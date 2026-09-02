# Using the Output Embedding to Improve Language Models

## Press & Wolf, EACL 2017 --- 中文学习笔记

> 论文：Ofir Press & Lior Wolf, **Using the Output Embedding to Improve
> Language Models**\
> 核心主题：**Weight Tying（权重共享 / 权重绑定）**

------------------------------------------------------------------------

# 0. 一句话理解这篇论文

语言模型原本通常有两张"词表"：

1.  **Input Embedding**：把输入 token 映射成向量；
2.  **Output Embedding / Output Projection**：把 hidden state 映射回
    vocabulary logits，用来预测下一个 token。

这篇论文发现：

> **输出层的权重本身也是一种很好的 word
> embedding，因此没必要维护两套独立参数，可以让输入 embedding 和输出
> embedding 共用同一套权重。**

也就是：

``` text
Input Embedding Weight
        ↓
        E
        ↑
Output Projection Weight
```

这就是 **Weight Tying**。

主要收益：

-   参数量更少；
-   减少 overfitting；
-   language model perplexity 往往更低；
-   在机器翻译中，可以显著减少模型大小而基本不损失性能。

这篇 2017 年论文的思想后来成为现代 Transformer / LLM 中非常常见的设计。

------------------------------------------------------------------------

# 1. 先理解传统 Language Model

假设：

``` text
Vocabulary size = V
Embedding dimension = D
```

一句话：

``` text
I love cats
```

训练时可能做：

``` text
Input:  I love
Target: cats
```

## Step 1：Input Embedding

token `love` 先经过 embedding matrix：

``` text
token id
   ↓
Embedding Matrix E_in
   ↓
embedding vector
```

矩阵：

``` text
E_in ∈ R^(V × D)
```

每一个 token 对应矩阵中的一行。

例如：

``` text
cat  → [0.2, 0.8, ...]
dog  → [0.3, 0.7, ...]
car  → [0.9, 0.1, ...]
```

语义相似的词通常会学到相似的向量。

------------------------------------------------------------------------

# 2. 模型最后还有另一张大矩阵

LSTM / Transformer 得到 hidden state：

``` text
h ∈ R^D
```

现在模型必须回答：

> vocabulary 中哪个 token 最可能是下一个？

所以需要把：

``` text
D-dimensional hidden state
```

转换成：

``` text
V-dimensional logits
```

传统模型使用：

``` text
logits = E_out @ h
```

其中：

``` text
E_out ∈ R^(V × D)
```

然后：

``` text
probabilities = softmax(logits)
```

得到：

``` text
P(cat)
P(dog)
P(car)
...
```

------------------------------------------------------------------------

# 3. 论文最重要的观察

注意：

``` text
E_in  ∈ R^(V × D)
E_out ∈ R^(V × D)
```

它们：

-   shape 一样；
-   都是一行对应一个 vocabulary token；
-   都在学习 word/token representation。

但是传统做法却是：

``` text
E_in != E_out
```

也就是训练两套完全独立的参数。

作者问：

> 为什么不能让它们是同一张 matrix？

于是提出：

``` text
E_in = E_out = E
```

这就是：

# Weight Tying

------------------------------------------------------------------------

# 4. Untied vs Tied

## Untied model

``` text
token
  ↓
E_in
  ↓
embedding
  ↓
LSTM / Neural Network
  ↓
hidden state h
  ↓
E_out
  ↓
logits
  ↓
softmax
```

需要两套参数：

``` text
E_in  : V × D
E_out : V × D
```

总共：

``` text
2 × V × D
```

------------------------------------------------------------------------

## Weight-Tied Model

``` text
                 ┌─────────────┐
token ─────────→ │             │
                 │      E      │
hidden state ──→ │             │
                 └─────────────┘
```

同一个矩阵同时负责：

``` text
Input:
token → embedding

Output:
hidden state → vocabulary logits
```

只需要：

``` text
V × D
```

参数直接少掉一大块。

------------------------------------------------------------------------

# 5. 一个非常小的数字例子

假设：

``` text
Vocabulary = ["I", "love", "cats", "dogs"]
V = 4

Embedding dimension D = 3
```

Embedding matrix：

``` text
E =

I      [0.1, 0.2, 0.3]
love   [0.4, 0.5, 0.6]
cats   [0.7, 0.8, 0.9]
dogs   [0.6, 0.8, 0.8]
```

输入：

``` text
cats
```

Input embedding 就是：

``` text
E["cats"]
=
[0.7, 0.8, 0.9]
```

经过模型以后得到：

``` text
h = [0.65, 0.75, 0.85]
```

预测下一个词时：

``` text
logit("I")    = E["I"]    · h
logit("love") = E["love"] · h
logit("cats") = E["cats"] · h
logit("dogs") = E["dogs"] · h
```

也就是说：

> 同一个 token vector 既用于"理解这个 token"，又用于判断 hidden state
> 和哪个候选 token 最匹配。

------------------------------------------------------------------------

# 6. 为什么 Output Weight 也可以被看成 Embedding？

Output layer 通常写成：

``` text
logits = W_out h
```

对 token `cat`：

``` text
logit_cat = W_cat · h
```

这里 `W_cat` 就是 output matrix 中属于 `cat` 的那一行。

如果：

``` text
W_cat ≈ h
```

dot product 通常较大：

``` text
W_cat · h ↑
```

模型就会认为：

``` text
P(cat | context) ↑
```

因此每个 output row 实际上也代表：

> "什么样的 hidden representation 应该对应这个词？"

所以它自然也是一种 token representation。

这就是论文题目：

> **Using the Output Embedding to Improve Language Models**

------------------------------------------------------------------------

# 7. 为什么 Weight Tying 可能让模型更好？

## 原因 1：减少参数

原本：

``` text
Input embedding  = V × D
Output embedding = V × D
```

Weight tying 后：

``` text
Shared embedding = V × D
```

少了一整个 vocabulary-sized matrix。

对于大 vocabulary，这非常可观。

------------------------------------------------------------------------

## 原因 2：Regularization

两套独立 embedding：

``` text
E_in
E_out
```

都有自己的自由度。

共享以后：

``` text
E_in = E_out
```

模型受到一个 constraint：

> 一个 token 的 representation 必须同时适合 input 和 output 两种任务。

这降低了模型随意 overfit 的空间。

论文实验发现，weight tying 往往：

``` text
training perplexity 可能不下降
validation/test perplexity ↓
```

这正是 regularization 的典型表现。

------------------------------------------------------------------------

# 8. 论文最有意思的 Gradient 观察

这是理解论文的关键。

## Untied Input Embedding

假设当前输入 token 是：

``` text
cat
```

那么一次训练 step 中，input embedding 主要更新：

``` text
E_in["cat"]
```

其他 token：

``` text
E_in["dog"]
E_in["car"]
E_in["apple"]
...
```

不会因为这个 input token 而直接更新。

所以：

> **Input embedding 的更新是 sparse 的。**

尤其 rare word：

``` text
rare token appears rarely
        ↓
its input embedding gets few updates
```

------------------------------------------------------------------------

## Output Embedding 不一样

Softmax 会计算整个 vocabulary：

``` text
P(word_1)
P(word_2)
...
P(word_V)
```

Cross entropy gradient 会影响 output matrix 中的许多/所有 rows。

因此：

> **Output embedding 在每个 training step 都获得更广泛的更新。**

论文写出的核心结论可以直觉化成：

``` text
Input embedding:
only current input row receives direct embedding update

Output embedding:
all vocabulary rows receive output-side updates
```

------------------------------------------------------------------------

# 9. Tied Embedding 的 Gradient

现在：

``` text
E_in = E_out = E
```

同一个矩阵同时承担两种角色。

因此：

``` text
gradient(E)
=
gradient_from_input
+
gradient_from_output
```

结果：

> shared embedding 每一步都可以从 output prediction 中获得训练信号。

论文进一步发现：

> **Tied embedding 最终的行为更像原本的 output embedding，而不是原本的
> input embedding。**

这是论文的重要理论分析之一。

------------------------------------------------------------------------

# 10. 为什么这可能帮助 rare words？

Untied input embedding：

``` text
rare word
   ↓
出现次数少
   ↓
input vector 更新次数少
```

Tied：

``` text
rare word vector
      ↓
不仅作为 input embedding
      ↓
还参与 vocabulary output prediction
      ↓
获得更多 training signal
```

因此参数利用更加充分。

------------------------------------------------------------------------

# 11. Perplexity 是什么？

论文大量使用：

``` text
Perplexity (PPL)
```

它是 language model 常用指标。

直觉：

> 模型预测下一个 token 时有多"不确定"。

**越低越好。**

例如：

``` text
Model A PPL = 100
Model B PPL = 70
```

通常：

``` text
Model B better
```

论文发现 weight tying 在多个 NNLM 实验中降低 validation/test
perplexity。

例如论文 PTB large model 的结果：

``` text
Baseline test PPL:      78.4
+ Weight Tying:         74.3
```

参数量同时：

``` text
66M → 51M
```

也就是说：

``` text
smaller model
+
better generalization
```

------------------------------------------------------------------------

# 12. Projection Regularization

作者还提出一个额外技巧：

``` text
hidden state h
     ↓
Projection P
     ↓
Output Embedding
```

即：

``` text
logits = E P h
```

并对 projection matrix `P` 做 regularization。

论文把它称为：

``` text
Projection Regularization (PR)
```

直觉：

Weight tying 强迫 input/output 完全共享。

Projection layer 给 output side 一点额外的 adaptation 能力：

``` text
shared representation
        +
small controlled transformation
```

同时通过 regularization 防止它过度自由。

在论文的小模型、无 dropout 实验中：

``` text
WT + PR
```

效果尤其明显。

------------------------------------------------------------------------

# 13. Word2Vec 为什么不一样？

论文还测试了 word2vec。

一个重要结果：

> Weight tying 并不是所有模型都有效。

对于 skip-gram word2vec：

``` text
Input embedding
```

和：

``` text
Output/context embedding
```

承担的角色不同。

实验中直接 tying：

``` text
Input = Output
```

反而使 embedding quality 下降。

因此不能得出：

> "所有 input/output matrices 都应该共享。"

正确结论是：

> **对于 Neural Language Models，input/output embedding sharing
> 特别有效。**

------------------------------------------------------------------------

# 14. Neural Machine Translation 中的 Weight Tying

传统 NMT 有至少三张 embedding：

``` text
Encoder Input Embedding
        ↓
source language

Decoder Input Embedding
        ↓
target language

Decoder Output Embedding
        ↓
target vocabulary prediction
```

作者首先共享：

``` text
Decoder Input
=
Decoder Output
```

叫：

``` text
Decoder Weight Tying
```

------------------------------------------------------------------------

# 15. Three-Way Weight Tying

使用 BPE/subword 后，源语言和目标语言会共享很多 subwords。

例如：

``` text
English:
international

French:
international
```

以及很多数字、名字、符号、subword pieces。

因此作者进一步提出：

``` text
Encoder Input
      =
Decoder Input
      =
Decoder Output
```

即：

# Three-Way Weight Tying (TWWT)

论文实验中，翻译模型参数量大幅下降，同时 BLEU 基本保持。

例如 EN→FR：

``` text
Baseline:
168M parameters
BLEU test = 33.13

Three-way weight tying:
80M parameters
BLEU test = 33.46
```

参数：

``` text
168M → 80M
```

减少超过一半，但翻译质量没有明显下降。

------------------------------------------------------------------------

# 16. 论文实验结论总结

作者研究了：

``` text
1. word2vec
2. Neural Network Language Models
3. Neural Machine Translation
```

主要发现：

### Word2Vec

``` text
Output embedding 本身质量不错
但 input/output weight tying 不合适
```

### Neural Language Models

``` text
Output embedding 比 input embedding 更好
Weight tying improves perplexity
Tied embedding 更像 output embedding
参数更少
generalization 更好
```

### Neural Machine Translation

``` text
Decoder weight tying
+
Three-way weight tying

可以显著减少参数
而基本不损害 translation quality
```

------------------------------------------------------------------------

# 17. 和现代 Transformer / LLM 的关系

虽然论文使用的主要语言模型是 LSTM/RNN
时代的模型，但思想可以直接映射到现代 Decoder-only Transformer。

现代 LM：

``` text
token ids
   ↓
Token Embedding
   ↓
Transformer Blocks
   ↓
Final Hidden State
   ↓
LM Head
   ↓
Vocabulary Logits
```

如果不 weight tying：

``` python
self.token_embedding = nn.Embedding(vocab_size, d_model)
self.lm_head = nn.Linear(d_model, vocab_size)
```

有两套权重。

Weight tying：

``` python
self.lm_head.weight = self.token_embedding.weight
```

概念上：

``` text
Token Embedding Weight
        ║
        ║ SAME PARAMETERS
        ║
LM Head Weight
```

注意 `nn.Linear` 的内部 weight shape 通常也是：

``` text
[vocab_size, d_model]
```

因此可以直接共享。

------------------------------------------------------------------------

# 18. 为什么对 LLM 参数量尤其重要？

假设：

``` text
Vocabulary V = 50,000
Hidden dimension D = 4096
```

一张 embedding：

``` text
50,000 × 4,096
≈ 205 million parameters
```

如果 input/output 分开：

``` text
≈ 410M parameters
```

Weight tying：

``` text
≈ 205M parameters
```

直接省掉约：

``` text
205M parameters
```

所以对于 vocabulary 很大的 language model，这不是一个微不足道的优化。

------------------------------------------------------------------------

# 19. 面试 / 考试版本

## Q: What is weight tying in language models?

**Answer:**

Weight tying means sharing the same parameter matrix between the input
token embedding and the output vocabulary projection.

Instead of learning:

``` text
E_input
W_output
```

separately, we enforce:

``` text
E_input = W_output
```

This reduces the number of parameters and can improve generalization and
perplexity.

------------------------------------------------------------------------

## Q: Why does it work?

三个关键词：

``` text
1. Parameter sharing
2. Regularization
3. Better use of training signal
```

更完整：

> Input and output embeddings both represent vocabulary items. Sharing
> them reduces redundant parameters and constrains the model to learn a
> representation useful for both encoding tokens and predicting tokens.
> The shared embedding also receives gradient signals from both roles.

------------------------------------------------------------------------

## Q: What did Press & Wolf find?

``` text
1. Output weights themselves form meaningful embeddings.
2. In NNLMs, output embeddings can be better than input embeddings.
3. Input/output weight tying improves language-model perplexity.
4. The tied embedding behaves more like the untied output embedding.
5. Weight tying substantially reduces model parameters.
6. In NMT, three-way tying can cut parameter count by more than half without hurting translation quality.
```

------------------------------------------------------------------------

# 20. 最值得记住的一张图

``` text
                 Traditional LM

token
  │
  ▼
┌─────────────┐
│ Input Emb.  │  E_in
└─────────────┘
  │
  ▼
Neural Network / Transformer
  │
  ▼
hidden state h
  │
  ▼
┌─────────────┐
│ Output Emb. │  E_out
└─────────────┘
  │
  ▼
softmax
  │
  ▼
next token


              Weight-Tied LM

token
  │
  ▼
┌──────────────────┐
│ Shared Embedding │◄──────────────┐
└──────────────────┘               │
  │                                │
  ▼                                │
Neural Network / Transformer       │
  │                                │
  ▼                                │
hidden state                       │
  │                                │
  └────────────────────────────────┘
          same weights
  │
  ▼
logits → softmax → next token
```

------------------------------------------------------------------------

# 21. 最终记忆口诀

> **进来的 token 要 embedding，出去的 prediction 也需要一张 vocabulary
> matrix。既然两边都在表示同一套 vocabulary，就让它们共享参数。**

或者更短：

``` text
Input embedding
       =
Output LM head

→ fewer parameters
→ regularization
→ often lower perplexity
```

------------------------------------------------------------------------

# 22. 这篇论文真正重要的思想

不要只把它记成一个 PyTorch trick：

``` python
lm_head.weight = embedding.weight
```

更重要的是它体现了一个很经典的 ML 思维：

> **如果模型中的两个参数在学习高度相关的结构，能否通过 parameter sharing
> 加入一个合理的 inductive bias？**

这样做可能同时获得：

``` text
更少参数
+
更好的泛化
+
更高的数据利用效率
```

Weight tying 就是一个非常漂亮的例子。

------------------------------------------------------------------------

## Paper Reference

Press, O., & Wolf, L. (2017). *Using the Output Embedding to Improve
Language Models*. Proceedings of EACL 2017, pp. 157--163.



# Weight Tying：Gradient 与参数更新机制

## 1. 核心结论

> **Weight Tying 不是"使用 Attention 更新后的 weight"。**

Input Embedding 和 Output Projection / LM Head 从头到尾共享同一个
parameter matrix `E`。

在一次 training step 的 forward pass 中，`E` 会被使用多次，但 Attention
不会直接修改 `E`。参数真正发生更新是在 `optimizer.step()`。

## 2. 一次 Training Step

假设：

``` text
Input:  "I love"
Target: "cats"
```

### Step 1：Input Embedding 使用 E

``` text
"love"
   ↓
E["love"]
   ↓
embedding
   ↓
Attention / FFN / Transformer
   ↓
hidden state h
```

此时 `E` 没有被修改。Forward pass 中 parameters are **used, not
updated**。

### Step 2：Output Projection 再次使用同一个 E

Weight Tying 规定：

``` text
W_output = E
```

因此：

``` text
h
↓
h @ E.T
↓
logits
↓
softmax
↓
P(next token)
```

同一个 `E` 在一次 forward 中承担两个角色：

``` text
E[token] → Input Embedding
h @ E.T  → Output Projection / LM Head
```

## 3. Forward 时 E 不会更新

错误理解：

``` text
Input 使用 E_old
↓
Attention
↓
Attention 把 E 更新成 E_new
↓
Output 使用 E_new
```

**不是这样。**

正确理解：

``` text
              E_current
              /       \
             /         \
Input Embedding       Output Projection
      ↓                     ↑
Transformer ─────────→ hidden state
```

整个 forward pass 使用同一个当前版本的 `E`。

## 4. Backpropagation

计算 loss 后：

``` python
loss.backward()
```

因为 `E` 在 forward 中存在两条计算路径，所以会收到两部分 gradient：

``` text
gradient(E)
=
gradient_from_input
+
gradient_from_output
```

PyTorch 会把它们累积到：

``` python
E.grad
```

直觉图：

``` text
                     Loss
                   ↙      ↘
          Output Path    Transformer
               ↓             ↓
        gradient_output  gradient_input
                   ↘      ↙
                       E
```

## 5. optimizer.step() 才真正更新 E

`loss.backward()` 只是计算 gradient，参数本身还没改变。

真正修改 parameter：

``` python
optimizer.step()
```

简单理解：

``` text
E_new
=
E_old - learning_rate × gradient(E)
```

而：

``` text
gradient(E)
=
gradient_input + gradient_output
```

所以：

``` text
E_new
=
E_old
-
lr × (gradient_input + gradient_output)
```

下一次 training step 才开始使用 `E_new`。

## 6. 为什么说 Shared Embedding 获得两边的 Training Signal？

``` text
E
├── Input Embedding role
│        ↓
│   gradient_input
│
└── Output Projection role
         ↓
    gradient_output
```

最终：

``` text
E.grad
=
gradient_input
+
gradient_output
```

因此同一个 embedding 同时学习：

``` text
How should this token be represented as an input?

+

What representation should correspond to predicting this token?
```

这就是：

> **The shared embedding receives training signals from both the input
> and output roles.**

## 7. 类比

把 `E` 想象成一个人，同时做两份工作：

``` text
Job A: Input Embedding
Job B: Output Projection
```

两个老板分别给 feedback：

``` text
Boss A → gradient_input
Boss B → gradient_output
```

总 feedback：

``` text
gradient_total
=
gradient_input + gradient_output
```

然后：

``` text
E_old
↓
optimizer.step()
↓
E_new
```

第二天两份工作都使用新的 `E`。

不是第一份工作结束后立刻更新，再让第二份工作使用新版本。

## 8. PyTorch Training Loop 必记顺序

``` text
① Forward
   Parameters are USED, not modified
        ↓
② Calculate Loss
        ↓
③ loss.backward()
   Calculate / accumulate gradients
   Parameters still NOT modified
        ↓
④ optimizer.step()
   Actually UPDATE parameters
        ↓
⑤ Next Training Step
   Use updated parameters
```

对应代码：

``` python
optimizer.zero_grad()

logits = model(x)       # Forward
loss = criterion(logits, target)

loss.backward()         # Calculate gradients
optimizer.step()        # Update parameters
```

## 9. Weight Tying 完整流程

``` text
                     Shared E
                    /        \
                   /          \
        Input Embedding       |
              ↓               |
         Transformer          |
              ↓               |
        Hidden State          |
              ↓               |
        Output Projection ←───┘
              ↓
            Loss
              ↓
           backward
              ↓
      ┌───────┴────────┐
      ↓                ↓
gradient_input   gradient_output
      └───────┬────────┘
              ↓
            E.grad
              ↓
       optimizer.step()
              ↓
            E_new
```

## 10. 面试 / 考试总结

> **In weight tying, the input embedding and output projection share the
> same parameter matrix. During forward propagation the shared weights
> are only used, not updated. During backpropagation, gradients from
> both the input and output computational paths accumulate on the shared
> parameter, and `optimizer.step()` updates it once for the next
> training step.**

## 11. 最短记忆版

``` text
Weight Tying
=
Input Embedding Weight
=
Output LM Head Weight
```

训练：

``` text
Forward
↓
同一个 E 用两次
↓
Loss
↓
Backward
↓
input gradient + output gradient
↓
E.grad
↓
optimizer.step()
↓
E 更新一次
↓
Next Step
```

> **Attention 不负责更新 embedding weight；真正更新参数的是
> optimizer.step()。**
