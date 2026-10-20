# Lab 06 结构化输出

!!! info "本章对应"
    - 生产组件：DeepInfer structure_output（编译-运行两段式）
    - miniflow 代码：`miniflow/structured_output.py`、`miniflow/engine.py`
    - 前置阅读：[10 结构化输出](../guide/10-structured.md)

---

## 1. Thompson NFA：正则子集的编译器

`GrammarAutomaton.compile(pattern)` 实现正则子集（字面量 /
`[a-z]` 字符类 / `.` / `*` / `|` / `()`）的 Thompson 构造 ——
递归下降解析 + NFA 片段拼接，ε-转移串联结构与优先级。

运行时是**字符级状态集**推进（NFA 天然多态并行）：

```python
def _step_chars(self, states, text):
    for ch in text:                      # token 的每个字符
        nxt |= 转移(states, ch)          # 状态集整体推进
        states = eps_closure(nxt)
        if not states: break             # 中途空 = 违约
    return states

def token_feasible(self, token_str):     # token 是否合法后继
    return bool(self._step_chars(set(self.states), token_str))
```

**粒度问题的解法**（[guide/10 第 4 节](../guide/10-structured.md)）：
token 喂给字符级自动机、允许**部分消费**（状态停在中间）—— 词表里
`'▁r'` 这种「schema 前缀」token 因此合法。miniflow 的教学词表是
最难的形态（空格独立成 token `'▁'`），实现无法取巧。

!!! danger "踩坑实录：初始状态硬编码 0"
    Thompson 构造的状态编号**从 1 开始**（`new_state()` 先自增再返回），
    而 `_reset()` 曾硬编码初始状态 `{0}`：

    ```python
    # 错：状态 0 不存在于转移表 → closure({0}) = {0} → 永远空转
    self.states = self._eps_closure({0})
    # 任何 pattern 的第一步 mask 就报「语法死局」
    ```

    修复：初始状态显式传入 —— `cls(trans, accept,
    start_state=node.start)`。**隐式的编号起点约定是状态机经典陷阱**：
    显式传递比隐式假设可靠一个数量级（这个 bug 让 SO demo 输出
    空串直接 aborted，而 mask 观测段直接 ValueError）。

## 2. 引擎集成：三个边界语义

`miniflow/engine.py` 的 `_ordinary_step` 处理 SO 请求的三个边界：

```python
# 1) 正常收笔：语法完整匹配且无合法后继
if automaton.in_accept() and not automaton.has_feasible_token(...):
    seq.finish(SequenceStatus.FINISHED_STOPPED)    # 不是全 -inf 喂采样

# 2) 死局 abort：词表与语法无交集
try:
    last = self._so_mask(automaton, last)
except ValueError:
    seq.finish(SequenceStatus.FINISHED_ABORTED)    # 明确失败

# 3) 掩码顺序：先屏蔽特殊 token（配合防注入），再施加语法掩码
masked[:_N_SPECIAL] = -np.inf
masked = automaton.mask(masked, self.tokenizer.vocab)
```

掩码在采样**前**施加 —— softmax 在掩码后归一化，合法 token 的相对
概率完全保留（约束不扭曲分布，[guide/10 第 3 节](../guide/10-structured.md)）。

`_N_SPECIAL = 5` 对应词表前 5 位特殊 token（bos/eos/pad + 两个控制位）
—— 与 [guide/11](../guide/11-recipe.md) 的 deepseek-recipe 防注入
构成双保险。

## 3. 词表事实：BPE 的空格是 `'▁'`

教学 BPE（`miniflow/tokenizer.py`）里空格是**独立 token `'▁'`**，
`ran`/`flew`/`in`/`over`/`the` 是各自独立的 token —— 「 ran」要
`['▁', 'ran']` 两个 token 拼出。所以 pattern 必须写成：

```python
PATTERN = "(▁ran|▁flew)(▁in|▁over)▁the"     # 字符级自动机的输入
TEXT_EQUIV = r"( ran| flew)( in| over) the"  # decode 后的等价正则
```

初始状态下唯一合法 token 是 `'▁'`（其它任何 token 的首字符都走不进
pattern 的第一个分支）—— demo 的 mask 观测段直接展示这一点。

!!! danger "踩坑实录：decode 的 strip() 吞前导空格"
    tokenizer.decode 早期带 `.strip()` —— SO 输出 `'▁ran…'` 解码成
    `' ran…'` 后被 strip 成 `'ran…'`，模式匹配（要求前导空格）失败。
    修复：decode 不做 strip —— **前导 `'▁'` 承载「续写新单词」的语义**，
    结构化输出的匹配依赖它。

## 4. 动手实验

```bash
python examples/demo_structured.py
#   [自由 greedy]   output = ' sat on the the'  （finished_length_capped）
#   [SO 约束]       output = ' ran in the'       （finished_stopped）
#   输出严格匹配模式 '( ran| flew)( in| over) the': True
#   初始状态下合法 token: ['▁']   ← mask 的直接观测

python -m pytest tests/test_structured.py -q
#   模式匹配 / 干净收笔 / 约束改变行为（≠ 自由输出）/ 死局 abort
```

??? question "思考题"
    1. 把 pattern 改成 `(▁ran|▁flew)(▁in|▁over)▁the.*`（加 `.*`）：
       finish_reason 从 `finished_stopped` 变成什么？为什么？
       （提示：`.*` 让任何后继都合法 → `in_accept` 永真但
       `has_feasible_token` 也永真 → 走 max_tokens 截断）
    2. `advance` 返回 False（采出的 token 违约）在什么情况下发生？
       引擎此时为什么选 abort 而不是重采样？
       （提示：mask 已保证 argmax 合法 —— 违约只可能来自掩码/推进
       的不一致，是引擎 bug 的信号，fail-fast 优先）
    3. 如何把「合法 token 集合」从 O(V·|pattern|) 的逐 token 模拟
       加速？（提示：生产 xgrammar 预编译「状态 → 合法 token 位图」
       的转移表 —— 编译期把运行期工作做掉，正是两段式的意义）

| 上一章 |
|---|
| [Lab 05 推测解码](05-speculative.md) |