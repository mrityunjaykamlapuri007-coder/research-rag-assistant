import re
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from rag_chain import get_answer
from retriever import retrieve
import pandas as pd

test_cases = [

{"question": "What is the Curie temperature of pure CaBi2Nb2O9?",
 "ground_truth": [
  "The Curie temperature of pure CaBi2Nb2O9 is approximately 930 degrees C.",
  "The Tc of pure CBN is approximately 933 degrees C.",
  "The Curie temperature of pure CaBi2Nb2O9 is 935 degrees C.",
  "CaBi2Nb2O9 possesses an extremely high Curie temperature of approximately 930 degrees C."
 ]},

{"question": "What is the piezoelectric coefficient of pure CaBi2Nb2O9?",
 "ground_truth": [
  "The piezoelectric constant of CBN is relatively poor with a value of approximately 5 pC/N.",
  "The d33 of pure CBNLCN-2.5 ceramics is approximately 5 pC/N."
 ]},

{"question": "What are the optimum properties of CBNLCN-15 ceramics?",
 "ground_truth": [
  "CBNLCN-15 ceramics have optimum properties with d33 of approximately 13.1 pC/N and Tc greater than 900 degrees C.",
  "Ca0.85(Li0.5Ce0.25Nd0.25)0.15Bi2Nb2O9 ceramics had d33 of approximately 13.1 pC/N and Tc of approximately 900 degrees C."
 ]},

{"question": "What is the space group of CaBi2Nb2O9?",
 "ground_truth": [
  "The space group of CaBi2Nb2O9 is A21am.",
  "CBN ceramics crystallize in the A21am space group with orthorhombic structure."
 ]},

{"question": "How does Li, Ce, Nd doping affect orthorhombic distortion in CBN?",
 "ground_truth": [
  "Small quantity of Li, Ce, Nd doping less than 2.5 mol% increases orthorhombic distortion because of smaller ionic radii of doping ions.",
  "With increasing Li, Ce, Nd doping from 5 to 25 mol%, orthorhombic distortion obviously decreased because replacement of asymmetric A-site Bi3+ by symmetric Li+, Ce3+ and Nd3+ ions decreased the orthorhombic distortion."
 ]},

{"question": "What sintering temperature was used for CBNLCN ceramics?",
 "ground_truth": [
  "CBNLCN ceramics were sintered at 1050 degrees C for 2 hours.",
  "The green pellets were sintered at 1050 degrees C for 2 h after burning out the binder at 550 degrees C."
 ]},

{"question": "What is the activation energy of CBNLCN ceramics at low temperature?",
 "ground_truth": [
  "The activation energy Ea at low temperature is in the range of 0.91 to 1.05 eV for CBNLCN-2.5, 5, 10, and 15 ceramics, indicating electrical conduction predominated by oxygen vacancies.",
  "The Ea values at low temperature are found to be in the range 0.91 to 1.05 eV for CBNLCN ceramics."
 ]},

{"question": "What is the thermal stability of CBNLCN-15 ceramics after annealing at 850 degrees C?",
 "ground_truth": [
  "CBNLCN-15 ceramics retain 11.3 pC/N which is 86% of initial d33 values after annealing at 850 degrees C for 2 hours.",
  "After annealing at 850 degrees C for 2 h, the d33 value of CBNLCN-15 ceramics retains 11.3 pC/N which is 86% of its initial values."
 ]},

{"question": "What causes the improvement of piezoelectric properties in CBNLCN-15 ceramics?",
 "ground_truth": [
  "The improvement of piezoelectric properties in CBNLCN-15 ceramics is attributed to decreasing grain sizes and morphotropic phase boundary.",
  "Decreasing grain sizes and morphotropic phase boundary MPB of orthorhombic and tetragonal phase near 100x = 15 improve piezoelectric properties."
 ]},

{"question": "What is the value of m for CaBi2Nb2O9 in the Aurivillius structure?",
 "ground_truth": [
  "The value of m for CaBi2Nb2O9 is 2.",
  "CaBi2Nb2O9 is a typical Aurivillius-type material with m equal to 2."
 ]},

{"question": "What is the general structural formula of Aurivillius compounds?",
 "ground_truth": [
  "The structure of Aurivillius compounds can be described by the general formula (Bi2O2)2+(Am-1BmO3m+1)2-.",
  "The general formula of Aurivillius compounds is (Bi2O2)2+(Am-1BmO3m+1)2- where A is mono-, di-, or trivalent ion, B is a transition metal cation, and m is the number of octahedral layers."
 ]},

{"question": "What type of material are Aurivillius compounds?",
 "ground_truth": [
  "Aurivillius phase compounds are bismuth layered structural ferroelectrics possessing high Curie temperature and fatigue-free properties.",
  "Aurivillius compounds called bismuth layered structural ferroelectrics are a large family of important ferroelectrics."
 ]},

{"question": "Why is CBN a promising candidate for high temperature applications?",
 "ground_truth": [
  "CBN is a promising candidate for high temperature applications because it has an extremely high Curie temperature of approximately 930 degrees C.",
  "CaBi2Nb2O9 is a promising candidate for applications in high-temperature sensors due to its extremely high Curie temperature."
 ]},

{"question": "What is the highest known Curie temperature of CaBi2Nb2O9?",
 "ground_truth": [
  "CaBi2Nb2O9 has the highest-known Curie temperature of 943 degrees C among two-layer Aurivillius phase ceramics.",
  "CBN is prospective for high-temperature sensor applications because it has the highest-known Tc of 943 degrees C."
 ]},

{"question": "What are the properties of CNBN-0.1 ceramics?",
 "ground_truth": [
  "CNBN-0.1 has the best performance with Tc of 925 degrees C, d33 of 14.4 pC/N, and Pr of 6.0 uC/cm2.",
  "Ca0.9(NaBi)0.05Bi2Nb2O9 has optimum d33 of 14.4 pC/N, Tc of 925 degrees C, and Pr of 6.0 uC/cm2."
 ]},

{"question": "How does NaBi modification affect the Curie temperature of CBN ceramics?",
 "ground_truth": [
  "NaBi modification slightly reduces the Curie temperature of CBN ceramics and the Tc values decrease from 925 degrees C to 880 degrees C with increasing NaBi content.",
  "Curie temperature slightly reduces and remains high with increasing NaBi content, in the range of 880 to 935 degrees C.",
  "NaBi modification causes the Curie temperature of CBN ceramics to decrease from 935 degrees C to 880 degrees C with increasing NaBi content.",
  "The Tc of CBN ceramics decreases from 935 degrees C to 920 degrees C for CNBN-0.1 and further to 870 degrees C for CNBN-0.3 with NaBi modification."
 ]},

{"question": "What is the d33 of pure CBN in the NaBi modification study?",
 "ground_truth": [
  "The d33 of pure CBN ceramic is approximately 6 pC/N.",
  "Pure CBN ceramics have d33 of 7 pC/N before NaBi modification."
 ]},

{"question": "What is the thermal stability of NaBi modified CBN ceramics?",
 "ground_truth": [
  "NaBi modified CBN ceramics show good temperature stability with d33 retaining 96% of initial value for CNBN-0.1 up to 800 degrees C.",
  "The d33 value remains in the order of 81% for CBN and 96% for CNBN-0.1 over the temperature from room temperature to 800 degrees C.",
  "NaBi modified CBN ceramics possess excellent thermal stability which is conducive to high temperature annealing temperatures below 900 degrees C.",
  "The NaBi modification improves the thermal stability of CBN ceramics with Tc values slightly decreasing from 925 degrees C to 880 degrees C."
 ]},

{"question": "What crystal structure does Ca1-x(NaBi)0.5xBi2Nb2O9 have?",
 "ground_truth": [
  "Ca1-x(NaBi)0.5xBi2Nb2O9 ceramics have orthorhombic crystal structure with space group A21am.",
  "CNBN ceramics crystallize in the orthorhombic system with space group A21am."
 ]},

{"question": "Why are high temperature piezoelectric sensors important?",
 "ground_truth": [
  "High temperature piezoelectric sensors are needed for monitoring engine components and must endure temperatures above 500 degrees C.",
  "Piezoelectric accelerometers in aircraft and aerospace industries are required to operate closest to engine components and must endure high temperatures more than 500 degrees C."
 ]},

{"question": "What is the value of m for CaBi4Ti4O15 in the Aurivillius structure?",
 "ground_truth": [
  "The value of m for CaBi4Ti4O15 is 4.",
  "CaBi4Ti4O15 is a typical Aurivillius structure ferroelectric with m equal to 4."
 ]},

{"question": "What are the optimum properties of Nb-Mg co-doped CaBi4Ti4O15?",
 "ground_truth": [
  "The CaBi4Ti3.9(Nb2/3Mg1/3)0.1O15 ceramic has optimum d33 of 24 pC/N and Tc of 787 degrees C.",
  "CNM-0.1 ceramic has a high d33 value of 24 pC/N accompanied by a high Tc of 787 degrees C and good thermal stability."
 ]},

{"question": "What causes the enhancement of DC resistivity in Nb-Mg co-doped CBT ceramics?",
 "ground_truth": [
  "The enhancement of DC resistivity in Nb-Mg co-doped CBT is due to the synergy effect of donor Nb and acceptor Mg which reduces intrinsic oxygen vacancy concentration.",
  "Donor Nb5+ eliminates oxygen vacancies while acceptor Mg2+ forms defect dipoles, together reducing oxygen vacancy concentration and increasing resistivity."
 ]},

{"question": "What is the general formula of bismuth layered structure ferroelectrics?",
 "ground_truth": [
  "The general formula of BLSFs is (Bi2O2)2+(Am-1BmO3m+1)2- where Bi2O2 is fluorite-like layer and Am-1BmO3m+1 is pseudo-perovskite structure.",
  "BLSFs have the general formula (Bi2O2)2+(Am-1BmO3m+1)2- where m usually belongs to 1 to 6."
 ]},

{"question": "What is the thermal stability of CBNCW-0.075 ceramics?",
 "ground_truth": [
  "CBNCW-0.075 ceramic maintains d33 of 12.7 pC/N after annealing at 900 degrees C for 30 min, decreasing by only 7.97%.",
  "The d33 of CBNCW-0.075 is 12.7 pC/N at 900 degrees C, showing good high-temperature stability."
 ]},

{"question": "How does Cu/W doping affect NbO6 octahedral distortion?",
 "ground_truth": [
  "Cu/W ions doping increases the distortion of NbO6 octahedron at proper doping amount, which enhances polarization strength of CBN ceramics.",
  "Cu/W co-doped CBN ceramics increase the distortion of NbO6 octahedron and significantly enhance piezoelectric constant and remnant polarization."
 ]},

{"question": "What is the effect of Cu/W doping on dielectric loss of CBN?",
 "ground_truth": [
  "Cu/W doping reduces the dielectric loss of CBN ceramics because CuO reduces dielectric loss and Cu/W ions reduce oxygen vacancy concentration.",
  "The dielectric loss of CBNCW-0.075 is only 0.57%, lower than pure CBN, because CuO effectively reduces dielectric loss."
 ]},

{"question": "What is the crystal structure of Cu/W co-doped CBN ceramics?",
 "ground_truth": [
  "Cu/W co-doped CBN ceramics have a bismuth layered structure of m equal to 2 and single-phase of CBN.",
  "CBNCW ceramics have a single CBN crystal structure with bismuth layered plate-like grain structure."
 ]},

{"question": "What is the remanent polarization of CBNCW-0.075?",
 "ground_truth": [
  "The remanent polarization Pr of CBNCW-0.075 is 12.10 uC/cm2.",
  "When x = 0.075, the Pr is significantly increased from 3.12 to 12.10 uC/cm2."
 ]},

{"question": "What is the piezoelectric coefficient of BTNO-1LiEr ceramics?",
 "ground_truth": [
  "BTNO-1LiEr obtains the highest d33 of 10.6 pC/N among the BTNO-1LiLn series.",
  "BTNO-1LiEr ceramic showed the maximum d33 of 10.6 pC/N."
 ]},

{"question": "What is the band gap of Bi3TiNbO9?",
 "ground_truth": [
  "The band gap Eg of Bi3TiNbO9 is approximately 2.143 eV.",
  "The electronic energy band calculation shows Bi3TiNbO9 belongs to direct band gap semiconductor with band gap of 2.143 eV."
 ]},

{"question": "How does the Curie temperature change with lanthanide ion radius in BTNO ceramics?",
 "ground_truth": [
  "The Curie temperature of BTNO-1LiLn increases with the decrease of Ln3+ ion radii.",
  "With increasing lanthanide atomic number and decreasing ion radius, the tolerance factor t decreases and Tc increases."
 ]},

{"question": "What is the thermal stability of BTNO-1LiLn ceramics?",
 "ground_truth": [
  "All BTNO-1LiLn ceramics show very good temperature stability with d33 remaining well until 800 degrees C.",
  "BTNO-1LiLn ceramics maintain stable d33 values until 800 degrees C before depolarization begins."
 ]},

{"question": "What does d33 represent in piezoelectric ceramics?",
 "ground_truth": [
  "d33 represents the piezoelectric coefficient which measures the electric charge generated per unit applied mechanical force.",
  "d33 is the piezoelectric constant that characterizes the piezoelectric activity of ceramics."
 ]},

{"question": "What does Pr represent?",
 "ground_truth": [
  "Pr represents the remanent polarization which is the residual polarization remaining after removal of the electric field.",
  "Pr is the remanent polarization that measures the ferroelectric response of ceramics."
 ]},

{"question": "What are the advantages of BLSF materials over PZT?",
 "ground_truth": [
  "BLSF materials have advantages of high Curie temperature above 500 degrees C, good thermal stability, and are lead-free unlike PZT which has Tc below 400 degrees C.",
  "Compared to PZT-based ceramics with low Curie temperature below 400 degrees C, BLSF ceramics have high Curie temperature, good thermal stability, and high quality factor."
 ]},

{"question": "What is the role of A-site substitution in improving CBN properties?",
 "ground_truth": [
  "A-site substitution is an effective way to improve the piezoelectric properties of CBN especially co-substitution with alkali metal cations and rare-earth cations.",
  "A-site substitution with Li+ and rare-earth ions effectively improves d33 values of CBN ceramics by modifying crystal structure distortion."
 ]},

{"question": "What is the role of oxygen vacancies in CBN ceramics?",
 "ground_truth": [
  "Oxygen vacancies affect electrical conductivity and dielectric loss of CBN ceramics by increasing carrier concentration.",
  "Oxygen vacancies pin ferroelectric domain walls and increase conductivity, resulting in lower d33 and higher dielectric loss in CBN ceramics."
 ]},

{"question": "Why does excessive doping reduce piezoelectric properties of CBN?",
 "ground_truth": [
  "Excessive doping causes reduction of NbO6 octahedral distortion, increase of internal defects, and increased relaxation which is not conducive to ceramic polarization.",
  "When doping concentration is too high, the DC resistivity of CBN ceramics decreases which results in smaller polarization voltage and lower d33 value."
 ]},

{"question": "What is the main limitation of pure CBN piezoelectric performance?",
 "ground_truth": [
  "The main limitation of pure CBN is that orientation of spontaneous polarization is restricted to the a-b plane, hindering practical applications.",
  "The piezoelectric constant of CBN is relatively poor because the rotation of spontaneous polarization Ps is limited in two-dimensional orientation."
 ]},

{"question": "What are the structural components of Aurivillius compounds?",
 "ground_truth": [
  "Aurivillius compounds consist of Bi2O2 fluorite-like layers and Am-1BmO3m+1 pseudo-perovskite layers arranged alternately along the c-axis.",
  "The structure consists of perovskite-like slabs separated by bismuth oxide Bi2O2 layers."
 ]},

{"question": "What is the orthorhombic crystal system of CaBi2Nb2O9?",
 "ground_truth": [
  "CaBi2Nb2O9 has orthorhombic crystal structure with space group A21am.",
  "CBN ceramics belong to the orthorhombic crystal system with A21am space group."
 ]}

]


def normalize_units(text):
    text = re.sub(r'°C', 'degrees C', text)
    text = re.sub(r'◦C', 'degrees C', text)
    text = re.sub(r'μC/cm2', 'uC/cm2', text)
    text = re.sub(r'μC/cm²', 'uC/cm2', text)
    text = re.sub(r'Ω·cm', 'ohm cm', text)
    text = re.sub(r'Ω\s*cm', 'ohm cm', text)
    text = re.sub(r'10\^(\d+)', r'10 to the power \1', text)
    text = re.sub(r'10\s*\^(\d+)', r'10 to the power \1', text)
    text = re.sub(r'×\s*10', 'times 10', text)
    text = re.sub(r'tanδ', 'tan delta', text)
    text = re.sub(r'tan\s*δ', 'tan delta', text)
    text = re.sub(r'~(\d)', r'approximately \1', text)
    text = re.sub(r'≈(\d)', r'approximately \1', text)
    text = re.sub(r'>', 'greater than', text)
    text = re.sub(r'<', 'less than', text)
    return text


def best_rouge(scorer, answer, ground_truths):
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    norm_answer = normalize_units(answer)
    for gt in ground_truths:
        norm_gt = normalize_units(gt)
        s = scorer.score(norm_gt, norm_answer)
        if s['rouge1'].fmeasure > best["rouge1"]:
            best["rouge1"] = s['rouge1'].fmeasure
        if s['rouge2'].fmeasure > best["rouge2"]:
            best["rouge2"] = s['rouge2'].fmeasure
        if s['rougeL'].fmeasure > best["rougeL"]:
            best["rougeL"] = s['rougeL'].fmeasure
    return best


def run_bert_scores(answers, ground_truths_list):
    all_answers, all_gts, mapping = [], [], []
    for i, (ans, gts) in enumerate(zip(answers, ground_truths_list)):
        safe_ans = normalize_units(ans) if ans and ans.strip() else "no answer"
        for gt in gts:
            all_answers.append(safe_ans)
            all_gts.append(normalize_units(gt))
            mapping.append(i)

    _, _, F1 = bert_score(all_answers, all_gts, lang="en",
                          model_type="bert-base-uncased", verbose=False)
    scores = F1.tolist()

    best = [0.0] * len(answers)
    for score, i in zip(scores, mapping):
        if score > best[i]:
            best[i] = score
    return best


def best_semantic_sim(embedder, answer, ground_truths):
    norm_answer = normalize_units(answer)
    emb_ans = embedder.encode(norm_answer)
    best = 0.0
    for gt in ground_truths:
        norm_gt = normalize_units(gt)
        emb_gt = embedder.encode(norm_gt)
        sim = util.cos_sim(emb_ans, emb_gt).item()
        if sim > best:
            best = sim
    return best


def run_evaluation():
    questions, answers, contexts, ground_truths_list = [], [], [], []

    print("Generating answers...\n")

    for i, test in enumerate(test_cases):
        print(f"Question {i+1}/{len(test_cases)}: {test['question']}")
        try:
            answer, _ = get_answer(test["question"], [])
            chunks = retrieve(test["question"])
        except Exception as e:
            print(f"  Failed on question {i+1}: {e}")
            answer = ""
            chunks = []

        questions.append(test["question"])
        answers.append(answer if answer and answer.strip() else "no answer")
        contexts.append([c["text"] for c in chunks])
        ground_truths_list.append(test["ground_truth"])

    print("\nCalculating ROUGE scores...")
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for ans, gts in zip(answers, ground_truths_list):
        best = best_rouge(rouge, ans, gts)
        rouge1_scores.append(best["rouge1"])
        rouge2_scores.append(best["rouge2"])
        rougeL_scores.append(best["rougeL"])

    print("Calculating BERTScores...")
    bert_scores = run_bert_scores(answers, ground_truths_list)

    print("Calculating Semantic Similarity...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    sim_scores = []
    for ans, gts in zip(answers, ground_truths_list):
        sim_scores.append(best_semantic_sim(embedder, ans, gts))

    best_ground_truths = [gts[0] for gts in ground_truths_list]

    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "ground_truth": best_ground_truths,
        "rouge1": rouge1_scores,
        "rouge2": rouge2_scores,
        "rougeL": rougeL_scores,
        "bert_score": bert_scores,
        "semantic_similarity": sim_scores
    })

    print("\n--- Evaluation Results ---")
    print(df[["question", "rouge1", "rouge2", "rougeL", "bert_score", "semantic_similarity"]].to_string())

    df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")

    print("\n--- Average Scores ---")
    print(f"ROUGE-1:             {df['rouge1'].mean():.3f}")
    print(f"ROUGE-2:             {df['rouge2'].mean():.3f}")
    print(f"ROUGE-L:             {df['rougeL'].mean():.3f}")
    print(f"BERTScore F1:        {df['bert_score'].mean():.3f}")
    print(f"Semantic Similarity: {df['semantic_similarity'].mean():.3f}")


if __name__ == "__main__":
    run_evaluation()