import argparse
from evaluator.bleu import _bleu
from rouge import Rouge  # 需要 pip install rouge
from bert_score import score  # 需要 pip install bert-score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file", type=str, required=True)
    parser.add_argument("--golds_file", type=str, required=True)
    args = parser.parse_args()

    # 读取文件
    with open(args.preds_file, 'r', encoding='utf-8') as f:
        preds = [line.strip() for line in f]
    with open(args.golds_file, 'r', encoding='utf-8') as f:
        refs = [line.strip() for line in f]

    assert len(preds) == len(refs)

    # 1. Calculate BLEU-4
    # 论文复现使用 evaluator/bleu.py 逻辑
    # 输入格式需为 list of strings (raw text)
    # _bleu 函数通常接受 (refs_list_of_lists, preds_list)
    # 我们这里简单包装一下
    refs_list = [[r] for r in refs]  # _bleu 需要 references 是列表的列表
    # 注意：_bleu 需要分词后的字符串。如果 evaluator.bleu 内部没有分词，这里可能需要 nltk.word_tokenize
    # 假设 pred/ref 已经是空格分词过的（英文）
    bleu_score = _bleu(args.golds_file, args.preds_file)  # 如果 evaluator.bleu 支持文件路径
    # 如果 evaluator.bleu.py 比较复杂，建议直接使用 nltk
    # from nltk.translate.bleu_score import corpus_bleu
    # bleu_score = corpus_bleu(refs_list, [p.split() for p in preds]) * 100

    print(f"BLEU-4: {bleu_score}")

    # 2. Calculate ROUGE-L
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(preds, refs, avg=True)
        print(f"ROUGE-L: {rouge_scores['rouge-l']['f'] * 100:.2f}")
    except Exception as e:
        print(f"ROUGE-L calculation failed: {e} (Maybe empty predictions?)")

    # 3. Calculate BERTScore
    try:
        P, R, F1 = score(preds, refs, lang="en", verbose=True)
        print(f"BERTScore F1: {F1.mean().item() * 100:.2f}")
    except Exception as e:
        print(f"BERTScore calculation failed: {e}")


if __name__ == "__main__":
    main()