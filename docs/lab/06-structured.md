# Lab 06 结构化输出

!!! info "本章对应"
    - 生产组件：DeepInfer structure_output（编译-运行两段式）
    - miniflow 代码：`miniflow/structured_output.py`、`miniflow/engine.py`
    - 前置阅读：[10 结构化输出](../guide/10-structured.md)

guide 10 讲了两段式的架构（编译期自动机 + 运行期掩码）和「约束不扭曲分布」的数学。本章进代码：`GrammarAutomaton` 怎么把正则编译成 NFA、token 级掩码怎么在字符级自动机上工作、以及两个真实踩坑——一个状态机经典陷阱（初始状态硬编码），一个看似无害的 `strip()`。

## 1. Thompson NFA：正则子集的编译器

`GrammarAutomaton.compile(pattern)` 实现正则子集的 Thompson 构造：支持字面量、`[a-z]` 字符类、`.`、`*`、`|`、`()`。

构造的思路（[guide/10 第 2 节](../guide/10-structured.md) 补了 NFA 的直觉）：**每种语法构造对应一个小 NFA 片段，递归下降解析正则时把片段拼起来**。字面量 `a` 是「状态 1 --a--> 状态 2」；`A|B` 是两条 ε 分支（ε 转移 = 不消费字符的白跳）从同一个入口指向 A、B 两个片段；`A*` 是入口一条 ε 弯道绕过 A 再回到入口。解析器递归读 pattern，每一步返回「入口状态 + 出口状态」的片段，外层把它们串联——`new_state()` 每次分配新编号，保证片段间状态不冲突。

### 1.1 手工构造一个完整例子

把 `(▁ran|▁flew)` 的 NFA 画出来（`▁` 当一个字面量字符看待）——**建议拿笔在纸上跟着画**，这是把 Thompson 构造从「知道」变成「会做」的唯一途径：

```text
递归下降解析 "(▁ran|▁flew)"：

1. 最外层括号 → 解析 "▁ran|▁flew"
2. 遇到 "▁ran" —— 字面量序列：
   片段: 1 --▁--> 2 --r--> 3 --a--> 4 --n--> 5
3. 遇到 "|" —— 记住这是「或」结构
4. 遇到 "▁flew"：
   片段: 6 --▁--> 7 --f--> 8 --l--> 9 --e--> 10 --w--> 11
5. 组装 A|B（新分配入口 12、出口 13）：
        12 --ε--> 1          （进分支 A）
        12 --ε--> 6          （进分支 B）
        5  --ε--> 13         （A 出）
        11 --ε--> 13         （B 出）

最终 NFA（14 个状态，转移表里记录 a 边和 ε 边）：
  字符边: 1→2(▁) 2→3(r) 3→4(a) 4→5(n) 6→7(▁) 7→8(f) 8→9(l) 9→10(e) 10→11(w)
  ε 边:   12→1, 12→6, 5→13, 11→13
  初始状态集 = eps_closure({12}) = {12, 1, 6}
  接受状态: {13}（这个子表达式的出口）
```

然后**手工走一遍输入 `'▁r'`**（这是 `'▁ran'` 的前两个字符——部分消费的场景）：

```text
初始状态集 {12, 1, 6}（closure 已含两个分支的入口）

读 '▁'：字符转移 {12,1,6} --▁--> {2, 7}    closure 后还是 {2,7}
读 'r'：{2,7} --r--> {3}                   closure {3}

状态集非空 → token '▁r' 可行 ✓（停在 A 分支中途）
下一个字符 'a' 或 'f'（对 7 来说 f 已错过）……
```

注意读 `'▁'` 时**两个分支同时活着**（{2,7}）——`'▁'` 是两个分支的共同前缀！这正是 NFA 状态集合的精髓：**不预判走哪条路，两条都走**，让后续字符自然淘汰（读 `r` 时 B 分支的 7 没有出边，静默死亡）。**「分支」的语义由集合的演化天然表达**——没有回溯、没有 if-else 的路标。

运行时是**字符级状态集**推进（NFA 的多态性用「维护一个状态集合」实现）：

```python
def _step_chars(self, states, text):
    for ch in text:                      # token 的每个字符
        nxt |= 转移(states, ch)          # 状态集整体推进一格
        states = eps_closure(nxt)        # 沿 ε 边免费扩张
        if not states: break             # 中途为空 = 这条路走死了
    return states

def token_feasible(self, token_str):     # token 是否合法后继
    return bool(self._step_chars(set(self.states), token_str))
```

**粒度问题的解法**（[guide/10 第 5 节](../guide/10-structured.md)）：token 喂给字符级自动机、允许**部分消费**（状态停在中间）——词表里 `'▁r'` 这种「pattern 前缀」token 因此合法。miniflow 的教学词表是最难的形态（空格独立成 token `'▁'`），实现无法取巧成「单 token 字符串匹配」。

（`eps_closure` 每步都要跑一遍——它的实现是一个小 BFS（沿 ε 边扩张直到不动点）。为什么「每字符步进后都要 closure」？因为 ε 边可能在字符消费后才可达（比如 `A|B` 里 A 分支死后集合回不到 B——不需要！集合里 B 一直活着；但 `A*` 的回环 ε 边需要随时重新激活）。**closure 的时机是 NFA 模拟正确性的一部分**——放在步进后保证「任何 ε 可达的状态都被考虑」。

!!! danger "踩坑实录：初始状态硬编码 0"

    Thompson 构造的状态编号**从 1 开始**（`new_state()` 先自增再返回），
    而 `_reset()` 曾硬编码初始状态集合为 `{0}`：

    ```python
    # 错：状态 0 不存在于转移表 → closure({0}) = {0} → 永远空转
    self.states = self._eps_closure({0})
    # 任何 pattern 的第一步 mask 就报「语法死局」
    ```

    症状很有欺骗性：不是崩溃，是**每个 pattern 都在第一步死局**——
    逻辑上「一切正常」，就是任何模式都走不通。修复：初始状态显式
    传入（`cls(trans, accept, start_state=node.start)`）。

    **隐式的编号起点约定是状态机经典陷阱**：显式传递比隐式假设
    可靠一个数量级。0 起点还是 1 起点，语言和实现里两种都常见
    （C 数组 0 起、Lua/Rust 枚举习惯各异）——凡是「魔法数字」，
    都该问一句它从哪来。

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

边界一为什么必须显式处理：全 `-inf` 的 logits 喂给采样器，argmax 是未定义行为（平局任取）；「语法走完了」本身是合法完成，语义上就该 `FINISHED_STOPPED`。边界二同理：词表和语法无交集（模式要求字符 `z`、词表没有 `z`）时逐 token 掩码永远全空——明确 abort 优于静默产出垃圾。

掩码在采样**前**施加——softmax 在掩码后归一化，合法 token 的相对概率完全保留（约束不扭曲分布，[guide/10 第 4 节](../guide/10-structured.md)）。`_N_SPECIAL = 5` 对应词表前 5 位特殊 token（bos/eos/pad + 两个控制位）——与 [guide/11](../guide/11-recipe.md) 的 deepseek-recipe 防注入构成双保险：约束生成期间，伪造的控制标记拿不到任何采样概率。

## 3. 词表事实：BPE 的空格是 `'▁'`

教学 BPE（`miniflow/tokenizer.py`）里空格是**独立 token `'▁'`**（GPT-2 同款），`ran`/`flew`/`in`/`over`/`the` 各自独立——「 ran」要 `['▁', 'ran']` 两个 token 拼出。所以 pattern 必须按 token 流的真实字符写：

```python
PATTERN = "(▁ran|▁flew)(▁in|▁over)▁the"     # 字符级自动机的输入
TEXT_EQUIV = r"( ran| flew)( in| over) the"  # decode 后的等价正则
```

初始状态下唯一合法的 token 是 `'▁'`——其它任何 token 的首字符都走不进 pattern 的第一个分支。demo 的 mask 观测段直接展示这一点（合法 token 列表 = `['▁']`）。

!!! danger "踩坑实录：decode 的 strip() 吞前导空格"

    tokenizer.decode 早期带 `.strip()`——SO 输出 `'▁ran…'` 解码成
    `' ran…'` 后被 strip 成 `'ran…'`，模式匹配（要求前导空格）失败。

    修复：decode 不做 strip——**前导 `'▁'` 承载「续写新单词」的语义**。
    这个 bug 的教训比它看起来深：strip 是「输出更干净」的善意改动，
    但它**静默改变了对外的语义边界**——对结构化输出，前导空格是
    输出契约的一部分。改「看起来无害」的输出处理时，先问谁依赖
    这个输出。

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
       `has_feasible_token` 也永真 → 走 max_tokens 截断。）
    2. `advance` 返回 False（采出的 token 违约）在什么情况下发生？
       引擎此时为什么选 abort 而不是重采样？
       （提示：mask 已保证 argmax 合法——违约只可能来自掩码/推进
       的不一致，是引擎 bug 的信号，fail-fast 优先。）
    3. 如何把「合法 token 集合」的计算从 O(V·|pattern|) 的逐 token
       模拟加速？（提示：生产 xgrammar 预编译「状态 → 合法 token
       位图」的转移表——编译期把运行期工作做掉，正是两段式的意义，
       呼应 [guide/12 原则五](../guide/12-philosophy.md)。）
    4. 1.1 节的手工构造里，`(▁ran|▁flew)` 有 14 个状态。算一算
        完整 pattern `(▁ran|▁flew)(▁in|▁over)▁the` 总共多少状态
        （每段字面量 n 个字符 = n+1 个状态 + 或结构的 2 个新状态 +
        串联不加新状态）？对照代码 `compile()` 后的 `len(trans)` 验证
        你的手算。**构造的状态数是 O(pattern 长度) 的线性关系**——
        这就是「编译轻」（guide/10 说的几百 ms 是 JSON 级大 schema，
        小模式是微秒）的原因。
    5. 用 1.1 节的方法手工构造 `a*` 的 NFA，然后手推输入 `""`（空串）
       和 `"aaa"` 的状态集合演化。空串初始就在接受态吗（`A*` 的 ε
       弯道）？——这解释了为什么 `.*` 模式「永真」（思考题 1 的
       机理在 NFA 层的样子）。

## 5. 本章收获的自检

1. Thompson 构造的三个积木（字面量/连接/或——加 `*` 的回环）？「或」的 ε 分支怎么表达「不预判走哪条路」？
2. 状态集合推进的每步两动作（字符转移 + ε closure）——closure 为什么每步都要跑？
3. 部分消费（`'▁r'` 合法）和「走完不死」的判定——token 粒度错位的解法。
4. 两个踩坑的签名：初始状态硬编码（一切 pattern 第一步死局——「逻辑正常但全走不通」）vs strip() 吞前导空格（语义边界被「善意的清理」破坏）。
5. 三个边界语义：收笔（STOPPED）/ 死局（ABORTED）/ 掩码顺序（先特殊 token 后语法——防注入双保险）。

| 上一章 |
|---|
| [Lab 05 推测解码](05-speculative.md) |