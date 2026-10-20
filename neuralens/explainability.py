"""Experiment 09: inspect model scores, not model thought processes."""
import re
from .sentiment import LABELS, _baseline, _transformer


def explain_baseline(text):
    if not text or not text.strip():
        raise ValueError("Enter a non-empty sentence")
    if len(text)>1000:
        raise ValueError("Maximum 1000 characters")
    model=_baseline()
    vectorizer, classifier=model.steps[0][1], model.steps[1][1]
    features=vectorizer.transform([text]); names=vectorizer.get_feature_names_out()
    classes=classifier.classes_.tolist()
    scores=classifier.decision_function(features)[0]
    probabilities=classifier.predict_proba(features)[0]
    indices=features.indices
    contributions=[]
    for j in indices:
        amount=float(features[0,j]); by_class={label:round(amount*float(classifier.coef_[i,j]),6) for i,label in enumerate(classes)}
        contributions.append({"feature":str(names[j]),"tfidf":round(amount,6),"by_class":by_class})
    contributions.sort(key=lambda f:abs(f["by_class"][classes[int(probabilities.argmax())]]),reverse=True)
    return {"text":text,"prediction":classes[int(probabilities.argmax())],"probabilities":{label:round(float(p),5) for label,p in zip(classes,probabilities)},"classes":classes,"logits":{label:round(float(v),5) for label,v in zip(classes,scores)},"intercepts":{label:round(float(v),5) for label,v in zip(classes,classifier.intercept_)},"contributions":contributions,"method":"Exact sparse TF-IDF × logistic-regression coefficients for each class; feature contributions plus intercept reconstruct class logits.","warning":"Correlated n-grams overlap. Contributions reflect this linear classifier, not causal word importance or a model's reasoning."}


def explain_transformer(text,max_words=16):
    if not text or not text.strip():
        raise ValueError("Enter a non-empty sentence")
    if len(text)>1000:
        raise ValueError("Maximum 1000 characters")
    import torch
    tokenizer,model=_transformer()
    def run(s):
        encoded=tokenizer(s,return_tensors="pt",truncation=True,max_length=512)
        with torch.inference_mode():
            p=torch.softmax(model(**encoded).logits[0],dim=-1).tolist()
        return {k:float(v) for k,v in zip(LABELS,p)}
    original=run(text)
    predicted=max(original,key=original.get)
    spans=list(re.finditer(r"\S+",text))
    rows=[]
    for match in spans[:max_words]:
        modified=(text[:match.start()]+text[match.end():]).strip()
        new=run(modified) if modified else original
        rows.append({"word":match.group(),"start":match.start(),"end":match.end(),"delta_predicted_probability":round(original[predicted]-new[predicted],5),"prediction_without":max(new,key=new.get),"probability_without":round(new[predicted],5)})
    return {"text":text,"prediction":predicted,"probabilities":{k:round(v,5) for k,v in original.items()},"word_removal":rows,"truncated_word_count":max(0,len(spans)-max_words),"method":"Delete one whitespace-delimited word at a time and rerun the pretrained sentiment model.","warning":"Deletion alters grammar, tokenization and distribution; probability differences are sensitivity probes, not faithful causal explanations. Scores are not necessarily calibrated."}
