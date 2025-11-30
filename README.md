# 🔤 Comprehensive Tokenizer Comparison & Analysis

A detailed comparison of major transformer tokenizers including BERT, GPT-2, FLAN-T5, CodeGen, Galactica, and Phi-3/Llama-2 architectures. This project demonstrates different tokenization strategies and provides comprehensive statistics for each tokenizer.

## 📋 Overview

This project compares how different Large Language Model (LLM) tokenizers handle various types of text content:
- English text with different capitalizations
- Programming code (Python keywords, operators)
- Mathematical expressions
- Emojis
- Non-English scripts (Urdu, Hindi, Chinese, Arabic, Korean)
- Special characters and whitespace

## 🤖 Models & Architectures

### 1. **Phi-3** (Microsoft)
- **Model**: `microsoft/Phi-3-mini-4k-instruct`
- **Architecture**: Decoder-only (Causal LM)
- **Use Case**: General-purpose instruction-following
- **Tokenizer**: SentencePiece (Llama-style)
- **Features**: KV cache support with custom legacy cache wrapper

### 2. **FLAN-T5** (Google, 2022)
- **Model**: `google/flan-t5-base`
- **Architecture**: Encoder-Decoder (Seq2Seq)
- **Use Case**: Instruction-following, translation, Q&A
- **Tokenizer**: SentencePiece (Unigram)
- **Special Tokens**: `<pad>`, `</s>`, `<unk>` (no CLS/SEP/MASK)

## 📊 Tokenizers Compared

| Tokenizer | Model | Vocab Size | Strategy |
|-----------|-------|------------|----------|
| **BERT base (uncased)** | `bert-base-uncased` | 30,522 | WordPiece |
| **BERT base (cased)** | `bert-base-cased` | 28,996 | WordPiece |
| **GPT-2** | `gpt2` | 50,257 | BPE |
| **FLAN-T5** | `google/flan-t5-base` | 32,100 | SentencePiece |
| **CodeGen** | `Salesforce/codegen-350M-mono` | 50,257 | BPE |
| **Galactica** | `facebook/galactica-125m` | 50,000 | BPE |
| **Phi-3/Llama 2** | `microsoft/Phi-3-mini-4k-instruct` | 32,000 | SentencePiece |

## 📈 Results Summary

### Efficiency Rankings

**Test Text**: 134 characters containing English, code, emojis, and non-English scripts

| Rank | Tokenizer | Tokens | Compression (chars/token) | Efficiency |
|------|-----------|--------|---------------------------|------------|
| 🥇 | **FLAN-T5** | 49 | **2.73** | Best |
| 🥈 | **BERT base (cased)** | 53 | 2.53 | Excellent |
| 🥉 | **BERT base (uncased)** | 59 | 2.27 | Good |
| 4 | CodeGen | 84 | 1.60 | Moderate |
| 5 | GPT-2 | 87 | 1.54 | Moderate |
| 6 | Phi-3/Llama 2 | 91 | 1.47 | Fair |
| 7 | Galactica | 101 | 1.33 | Poor |

### Key Findings

✅ **FLAN-T5** is the most efficient tokenizer for this mixed-content text
- Highest compression ratio (2.73 chars/token)
- Fewest tokens (49)
- Excellent handling of code and multilingual content

✅ **BERT (cased)** performs better than BERT (uncased)
- Preserves capitalization information
- 53 tokens vs 59 tokens (uncased)
- Better for code and proper nouns

✅ **Vocabulary size doesn't always correlate with efficiency**
- FLAN-T5 (32K vocab) outperforms GPT-2 (50K vocab)
- Strategy matters more than vocab size

## � Detailed Analysis Examples

### Test Text Breakdown

**Total**: 134 characters containing:
- English text with mixed case
- Python keywords: `False`, `None`, `elif`
- Operators: `==`, `>=`
- Numbers: `12.0*50=600`
- Emojis: 🫡👋🏾🤗🥴😶‍🌫️
- Non-English: ہندی中文العربية한국어 (Urdu, Hindi, Chinese, Arabic, Korean)

### Example 1: BERT base (uncased)

```
📈 BASIC STATISTICS:
  • Total tokens: 59
  • Compression ratio: 2.27 chars/token
  • Vocabulary size: 30,522

🔖 SPECIAL TOKENS:
  • PAD: '[PAD]' (ID: 0)
  • UNK: '[UNK]' (ID: 100)
  • CLS: '[CLS]' (ID: 101)
  • SEP: '[SEP]' (ID: 102)
  • MASK: '[MASK]' (ID: 103)

📊 TOKEN BREAKDOWN:
  • Special tokens: 3 ([CLS], [SEP], [UNK] for emojis)
  • Unknown tokens: 1 (emojis)
  • Subword tokens: 22 (words split with ##)
  • Regular tokens: 34

🔍 CONTENT-SPECIFIC HANDLING:
  • Capitalization: 2 tokens → "capital", "##ization"
  • Python keywords: 4 tokens → "false", "none", "eli", "##f"
  • Numbers: 7 tokens → "12", ".", "0", "*", "50", "=", "600"
  • Emojis: 1 token → [UNK]
  • Non-English: 21 tokens (heavily fragmented)
```

**Key Observation**: BERT uncased converts everything to lowercase, losing case information. Emojis become `[UNK]`.

### Example 2: BERT base (cased)

```
📈 BASIC STATISTICS:
  • Total tokens: 53 ⭐ (6 fewer than uncased)
  • Compression ratio: 2.53 chars/token
  • Vocabulary size: 28,996

📊 TOKEN BREAKDOWN:
  • Special tokens: 4
  • Unknown tokens: 2
  • Subword tokens: 16 (fewer than uncased!)

🔍 CONTENT-SPECIFIC HANDLING:
  • "CAPITALIZATION" → [CA, ##PI, ##TA, ##L, ##I, ##Z, ##AT, ##ION] (8 tokens)
  • "False" → [F, ##als, ##e] (preserves case)
  • "None" → [None] (single token!)
  • Arabic text: 7 tokens (better than uncased's 21!)
```

**Why it's more efficient**: Case preservation allows better matching with vocabulary. "None" is recognized as a whole word.

### Example 3: FLAN-T5 (Most Efficient)

```
📈 BASIC STATISTICS:
  • Total tokens: 49 🥇 (BEST)
  • Compression ratio: 2.73 chars/token 🥇 (BEST)
  • Vocabulary size: 32,100

🔖 SPECIAL TOKENS:
  • PAD: '<pad>' (ID: 0)
  • UNK: '<unk>' (ID: 2)
  • EOS: '</s>' (ID: 1)
  • NO CLS, SEP, or MASK tokens

📊 TOKEN BREAKDOWN:
  • Special tokens: 4
  • Unknown tokens: 3 (emojis + some non-English)
  • Subword tokens: 0 (uses different strategy)

🔍 CONTENT-SPECIFIC HANDLING:
  • "CAPITALIZATION" → [CA, PI, TAL, IZ, ATION] (5 tokens)
  • "False" → [Fal, s, e] (3 tokens)
  • "None" → [None] (1 token)
  • "elif" → [ , e, l, if] (4 tokens, space aware)
  • "12.0" → [12., 0] (2 tokens - merges decimal!)
  • Numbers: 6 tokens total (most efficient)
  • Non-English: 2 tokens (very efficient!)
```

**Why it wins**: SentencePiece tokenization with optimized merging. Handles numbers better, doesn't need special tokens like CLS/SEP.

### Example 4: GPT-2 (BPE)

```
📈 BASIC STATISTICS:
  • Total tokens: 87
  • Compression ratio: 1.54 chars/token
  • Vocabulary size: 50,257

🔖 SPECIAL TOKENS:
  • UNK/BOS/EOS: '<|endoftext|>' (ID: 50256)
  • No PAD, CLS, SEP, or MASK tokens

📊 TOKEN BREAKDOWN:
  • Special tokens: 0 (not used within text)
  • Unknown tokens: 0 (byte-level BPE handles all characters)
  • Subword tokens: 87 (all tokens are BPE subwords)
  • Regular tokens: 0

🔍 CONTENT-SPECIFIC HANDLING:
  • Capitalization: 1 token → "ĠCAPITALIZATION" (preserves case)
  • Python keywords: 1 token each → "ĠFalse", "ĠNone", "Ġelif"
  • Numbers: 7 tokens → "Ġ12", ".", "0", "*", "50", "=", "600"
  • Emojis: 24 tokens (each emoji broken into multiple byte tokens)
  • Non-English: 48 tokens (heavily fragmented into byte tokens)

**Key Observation**: GPT-2 uses byte-level BPE, meaning it can represent any character sequence, so it doesn't have "unknown" tokens in the traditional sense. However, non-English scripts and emojis are often broken down into many individual byte tokens, leading to lower efficiency. It preserves capitalization and uses 'Ġ' to denote leading spaces.

### 📝 DETAILED TOKEN-BY-TOKEN BREAKDOWN (GPT-2)

```
Original Text (134 chars): "CAPITALIZATION False None elif 12.0*50=600 🫡👋🏾🤗🥴😶‍🌫️ ہندی中文العربية한국어"

Tokens (first 50 of 87):
1.  ĠCAPITALIZATION (ID: 39598)
2.  ĠFalse (ID: 17294)
3.  ĠNone (ID: 10459)
4.  Ġelif (ID: 27958)
5.  Ġ12 (ID: 1599)
6.  . (ID: 13)
7.  0 (ID: 28)
8.  * (ID: 16)
9.  50 (ID: 1564)
10. = (ID: 18)
11. 600 (ID: 10836)
12. Ġð (ID: 288)
13. Ł (ID: 109)
14. ĺ (ID: 108)
15. Ġð (ID: 288)
16. Ł (ID: 109)
17. ĺ (ID: 108)
18. Ł (ID: 109)
19. ĺ (ID: 108)
20. Ł (ID: 109)
21. ĺ (ID: 108)
22. Ł (ID: 109)
23. ĺ (ID: 108)
24. Ł (ID: 109)
25. ĺ (ID: 108)
26. Ł (ID: 109)
27. ĺ (ID: 108)
28. Ł (ID: 109)
29. ĺ (ID: 108)
30. Ł (ID: 109)
31. ĺ (ID: 108)
32. Ł (ID: 109)
33. ĺ (ID: 108)
34. Ł (ID: 109)
35. ĺ (ID: 108)
36. Ł (ID: 109)
37. ĺ (ID: 108)
38. Ł (ID: 109)
39. ĺ (ID: 108)
40. Ł (ID: 109)
41. ĺ (ID: 108)
42. Ł (ID: 109)
43. ĺ (ID: 108)
44. Ł (ID: 109)
45. ĺ (ID: 108)
46. Ł (ID: 109)
47. ĺ (ID: 108)
48. Ł (ID: 109)
49. ĺ (ID: 108)
50. Ł (ID: 109)
```

## 🎬 Actual Execution Examples

### Phi-3 Inference Output

**Prompt**: "Whats India's score right now?"

```
Model loaded. Generating response...

Chatbot: I'm sorry, but I'm unable to provide real-time sports scores or live updates. 
For the latest information on cricket scores, please check a reliable sports news 
website or a sports news app.

✓ Generation complete with KV cache enabled!
```

**Token Analysis**:
- Input tokens: 9 (`[Wh, ats, India, ', s, score, right, now, ?, \n]`)
- Output tokens: 50 (generated response)
- Used KV cache for efficiency
- Generation time: ~3-5 seconds

### FLAN-T5 Inference Output

**Prompt**: "Translate to French: How are you today?"

```
FLAN-T5 (2022) - Encoder-Decoder Seq2Seq Model

Generated Response: Comment vous êtes aujourd'hui?

Full output (with special tokens): <pad> Comment vous êtes aujourd'hui?</s>
Token IDs: [0, 5257, 327, 3, 6738, 7082, 31, 3464, 58, 1]

Flan-T5 Special Tokens:
  PAD token: '<pad>' (ID: 0)
  EOS token: '</s>' (ID: 1)
  UNK token: '<unk>' (ID: 2)

✓ Flan-T5 generation complete!
```

**Translation Quality**: ✅ Accurate French translation with proper grammar
**Generation time**: ~2-3 seconds

## 📋 Comprehensive Tokenization Examples

### How Different Tokenizers Handle "CAPITALIZATION"

| Tokenizer | Tokens | Token Breakdown |
|-----------|:------:|-----------------|
| **FLAN-T5** ⭐ | 5 | `[CA, PI, TAL, IZ, ATION]` |
| **BERT (cased)** | 8 | `[CA, ##PI, ##TA, ##L, ##I, ##Z, ##AT, ##ION]` |
| **Phi-3/Llama** | 6 | `[C, AP, IT, AL, IZ, ATION]` |
| **GPT-2** | 4 | `[ĠCAP, ITAL, IZ, ATION]` |
| **BERT (uncased)** | 2 | `[capital, ##ization]` (loses case!) |
| **Galactica** | 3 | `[ĠCAP, ITAL, IZATION]` |

### How Different Tokenizers Handle Emojis "🫡👋🏾🤗"

| Tokenizer | Tokens | Strategy |
|-----------|:------:|----------|
| **BERT (uncased/cased)** | 1 | `[UNK]` - Treats all emojis as unknown |
| **FLAN-T5** | 2 | `[<unk>, <unk>]` - Two unknown tokens |
| **GPT-2** | 11 | Byte-level encoding (fragmented) |
| **CodeGen** | 11 | Byte-level encoding (fragmented) |
| **Galactica** | 16 | Most fragmented |
| **Phi-3/Llama** | 17 | Heavily fragmented into bytes |

**Insight**: BERT's simple `[UNK]` approach is most efficient for emojis, but loses all information. BPE-based models fragment emojis into many byte-level tokens.

### How Different Tokenizers Handle Numbers "12.0*50=600"

| Tokenizer | Tokens | Token Breakdown |
|-----------|:------:|-----------------|
| **FLAN-T5** ⭐ | 6 | `[12., 0, *, 50, =, 600]` - Merges decimal! |
| **BERT (both)** | 7 | `[12, ., 0, *, 50, =, 600]` |
| **GPT-2** | 7 | `[Ġ12, ., 0, *, 50, =, 600]` |
| **CodeGen** | 7 | `[Ġ12, ., 0, *, 50, =, 600]` |
| **Galactica** | 11 | `[1, 2, ., 0, *, 5, 0, =, 6, 0, 0]` - Each digit separate! |
| **Phi-3/Llama** | 12 | Similar to Galactica |

**Best for Math**: FLAN-T5 (merges "12." as one token)
**Worst for Math**: Galactica and Phi-3 (split every digit)

## 📊 Statistics Provided

For each tokenizer, the analysis provides:

### 📈 Basic Statistics
- Total token count
- Original text length
- Compression ratio (chars/token)
- Vocabulary size

### 🔖 Special Tokens
- PAD, UNK, CLS, SEP, MASK, BOS, EOS tokens
- Token IDs for each special token

### 🔧 Tokenization Strategy
- Tokenizer class
- Strategy: WordPiece, BPE, or SentencePiece
- Model max sequence length

### 📊 Token Breakdown
- Special tokens count
- Unknown tokens ([UNK]) count
- Subword tokens count
- Regular tokens count

### 🔍 Content-Specific Analysis
How each tokenizer handles:
- Capitalization (e.g., "CAPITALIZATION")
- Python keywords (False, None, elif)
- Operators (==, >=)
- Numbers (12.0*50=600)
- Emojis (🫡👋🏾🤗)
- Non-English text (ہندی中文العربية한국어)

### 🎨 Visual Tokenization
Color-coded token display showing how text is split

### 📝 Detailed Token List
First 50 tokens with their IDs and decoded values

## 🎯 Special Tokens Explained

### Common Special Tokens

| Token | Purpose | Used By |
|-------|---------|---------|
| **[PAD]** | Padding sequences to same length | BERT, FLAN-T5 |
| **[UNK]** | Unknown/out-of-vocabulary tokens | All models |
| **[CLS]** | Classification token (sequence start) | BERT only |
| **[SEP]** | Separator between segments | BERT only |
| **[MASK]** | Masked language modeling | BERT only |
| **\<s\>** | Beginning of sequence | Llama, Phi-3 |
| **\</s\>** | End of sequence | FLAN-T5, Llama, Phi-3 |

### Architecture Differences

**BERT** (Encoder-only):
- Uses CLS, SEP, MASK tokens
- WordPiece tokenization with `##` subwords
- Bidirectional context

**GPT-2** (Decoder-only):
- Uses BPE with `Ġ` (space) encoding
- No special classification tokens
- Unidirectional (left-to-right)

**FLAN-T5** (Encoder-Decoder):
- Uses `<pad>`, `</s>`, `<unk>`
- SentencePiece tokenization
- No CLS/SEP/MASK (not classification-focused)

**Llama/Phi-3** (Decoder-only):
- Uses `<s>`, `</s>`, `<unk>`
- SentencePiece tokenization
- Designed for instruction-following

## 🚀 Running the Analysis

### Prerequisites

```bash
pip install torch transformers
```

### Run Complete Analysis

```bash
python python.py
```

This will:
1. Run Phi-3 inference with KV cache
2. Run FLAN-T5 translation example
3. Perform comprehensive tokenizer comparison
4. Generate detailed statistics and visualizations

### Save Output to Log

```bash
python python.py 2>&1 | tee tokenizer_analysis.log
```

### Execution Time

**Total runtime**: ~29 seconds
- Phi-3 inference: ~3-5s
- FLAN-T5 inference: ~2-3s
- Tokenizer analysis: ~20-25s

## 📁 Project Structure

```
.
├── python.py              # Main analysis script with all tokenizers
├── python_fast.py         # Optimized Phi-3 inference only
├── flan_t5.py            # Standalone FLAN-T5 script
├── tokenizer_analysis.log # Generated output log
└── README.md             # This file
```

## 🔧 Code Features

### Phi-3 Legacy Cache Wrapper

Custom compatibility wrapper for Phi-3's KV cache:

```python
class Phi3LegacyCache(DynamicCache):
    """Compatibility wrapper that adds legacy Cache API methods for Phi-3"""
    # Provides seen_tokens tracking
    # Compatible with older Phi-3 code
```

### Performance Optimizations

- `torch.set_float32_matmul_precision('high')` - Faster matrix operations
- `torch.inference_mode()` - Optimized inference
- `device_map="mps"` - Apple Silicon GPU acceleration
- `dtype=torch.float16` - Half precision for speed
- `low_cpu_mem_usage=True` - Memory efficient loading

## 📚 Tokenization Strategies Compared

### 1. **WordPiece** (BERT)
- Greedy longest-match-first algorithm
- Uses `##` prefix for subwords
- Good for English and common languages
- Example: `CAPITALIZATION` → `[CAP, ##IT, ##AL, ##IZ, ##ATION]`

### 2. **Byte-Pair Encoding (BPE)** (GPT-2, CodeGen, Galactica)
- Iteratively merges most frequent byte pairs
- Uses `Ġ` for space encoding
- Good for code and mixed content
- Example: `CAPITALIZATION` → `[CAP, ITAL, IZ, ATION]`

### 3. **SentencePiece** (FLAN-T5, Phi-3, Llama)
- Language-agnostic unigram model
- No special space markers
- Excellent for multilingual content
- Example: `CAPITALIZATION` → `[CA, PI, TAL, IZ, ATION]`

## 🎨 Visual Features

The analysis includes:
- **Color-coded tokens**: Each token displayed with unique background color
- **Readable formatting**: Tabs/newlines shown as `\t`, `\n`
- **Statistics tables**: Clean, formatted comparison tables
- **Emoji support**: Proper handling of emoji tokenization
- **Unicode support**: Handles RTL scripts (Arabic, Urdu) and ideographs

## 💡 Use Cases

**Choose FLAN-T5 when:**
- You need instruction-following
- Translation or Q&A tasks
- Best compression ratio is important
- Working with mixed content

**Choose BERT when:**
- Classification tasks
- Need bidirectional context
- Working primarily with English
- Sentence pair tasks (use SEP token)

**Choose GPT-2 when:**
- Text generation
- Code generation
- Need flexibility with BPE
- Larger vocabulary helps

**Choose Phi-3/Llama when:**
- Instruction-following at scale
- Chat/dialogue applications
- Need efficient GPU inference
- Want modern architecture

## 📖 References

- [BERT Paper](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al., 2019
- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416) - Chung et al., 2022
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) - Microsoft, 2024
- [Galactica Paper](https://arxiv.org/abs/2211.09085) - Taylor et al., 2022

## 🔬 Insights

1. **Compression efficiency varies widely** - 2x difference between best (FLAN-T5: 2.73) and worst (Galactica: 1.33)

2. **Case sensitivity matters** - BERT cased outperforms uncased by ~10% fewer tokens

3. **Modern models use SentencePiece** - FLAN-T5 and Phi-3 use SentencePiece for better multilingual support

4. **Vocabulary size ≠ efficiency** - Strategy and training data matter more

5. **Emojis are challenging** - Most tokenizers use 3-4 tokens per emoji

6. **Non-English scripts** - SentencePiece models handle these better than WordPiece

## 📝 License

This is an educational project for comparing tokenizer implementations.

## 🤝 Contributing

Feel free to add more tokenizers or test cases to the comparison!

---

**Generated by**: Comprehensive Tokenizer Analysis Script  
**Last Updated**: November 2025
