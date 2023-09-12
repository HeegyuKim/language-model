from korouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])

ref = "안녕하세요 반가워요 오늘 뭐 먹을래요?"
pred = "안녕 ㅋ 반가워 오늘 뭐 할래?"

print(scorer.score(ref, pred))