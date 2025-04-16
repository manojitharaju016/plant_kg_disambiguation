import pandas as pd

#read the kmeans clusters
df = pd.read_parquet('/path/to/your/kmeans_clusters/multisect_kmeans_all_clusters_max_clust_30.parquet',engine='pyarrow')

#remove rows where node_ids_in_cluster is only 1 by checking the length of the list
df = df[df['node_ids_in_cluster'].apply(lambda x: len(x) > 1)]


# convert the numpy arrays to lists
df['node_ids_in_cluster'] = df['node_ids_in_cluster'].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)


# randomly sample 100 clusters
df = df.sample(n=100, random_state=1).reset_index(drop=True)
#df

#store the above result in a list of dictionaries
entity_list = df.set_index('cluster_id')['node_ids_in_cluster'].to_dict()

# The system message and user message are provided as strings.
system_message = """You are a data scientist specializing in grouping plant biological entities. Your task is to cluster similar entities while strictly adhering to the following guidelines:
	1.	Exact Phrase Matching Matters: Always consider the full phrase, including key biological terms, bracketed text (ignoring minor differences such as spacing, punctuation, correct abbreviations, plurality).
	2.	Strict (100%) Key Term Separation: Entities with different biological terms MUST be placed in separate clusters.
3. Sub-identifier separation: Separate Entities with numeric differences, sub-identifiers, or qualifiers into different groups.
	4.	Avoid False Similarity: Do NOT cluster two items together in same group just because they share a common word or term.
	5.	Strict Synonym/Near-Synonym Grouping: Only group entities that refer to the same biological structure, process, meaning or concept.
	6.	Maintain 100% Precision: When in even small doubt, MUST place entities in separate clusters.
	7.	Preserve Original Data: No new items should be introduced, no duplicates should be introduced, and no entities should be omitted.
8. YOU MUST pickup most appropriate cluster representative and enclose it with '**', if there is more than one entity in that particular cluster.
	9.	Output Format: Always return results in valid JSON format. MUST USE GIVEN KEY.
"""

user_message = """Input-1
{
  "0": '['Meiotic block [phenotype]', 'early arrest [phenotype]', 'arrest of meiotic progression in anaphase II [phenotype]', 'arrested zygotic divisions [phenotype]', 'meiotic arrest [phenotype]', 'delayed/arrested meiosis [phenotype]', 'pachytene arrest [phenotype]', 'block in meiosis prophase I [phenotype]', 'male meiosis until the end of the first division [phenotype]', 'meiotic prophase arrest [phenotype]', 'arrest in late prophase I [phenotype]', 'arrest at the end of meiosis I [phenotype]', 'leptotene arrest [phenotype]', 'meiosis I arrest [phenotype]', 'arresting the first mitosis during gametogenesis [phenotype]', 'absence of meiotic arrest [phenotype]', 'meiotic arrest phenotype [phenotype]', 'meiotic arrest at telophase I [phenotype]', 'termination of meiosis after anaphase I [phenotype]', 'premature termination of meiosis after anaphase I [phenotype]', 'mitotic arrest during female gametogenesis [phenotype]', 'arrest after meiosis I [phenotype]', 'meiotic arrest at pachytene [phenotype]', 'arrested endosperm nuclear divisions [phenotype]', 'meiotic arrest in anaphase II [phenotype]', 'meiotic division stop [phenotype]', 'arrest of the first mitotic division in gametogenesis [phenotype]', 'FNM half-stop [phenotype]', 'males arresting in the middle of prophase I [phenotype]', 'arresting prior to the first mitotic division [phenotype]']'
}
Input-1 → Output-1 [REASONING]

1) Generic Meiotic Arrest

Cluster:
[ "Meiotic block [phenotype]", "**meiotic arrest [phenotype]**", "meiotic arrest phenotype [phenotype]", "meiotic division stop [phenotype]" ]
• Incorrect: Grouping these separately as if they refer to different stages would obscure their identical meaning.
• Correct: Recognize they all broadly indicate a total halt in meiosis at an unspecified stage.
• Important: These terms are functionally synonymous, capturing any generic failure of meiotic progression.
• Cluster representative: I selected 'meiotic arrest [phenotype]' as the representative because it is the most concise and commonly used term to describe a complete halt in the meiotic process. Unlike the other phrases which include additional descriptors or less formal wording, 'meiotic arrest' clearly and unambiguously captures the essential biological event, making it the best exemplar for the cluster.

2) Delayed / Arrested Meiosis

Cluster:
[ "delayed/arrested meiosis [phenotype]" ]
• Incorrect: Merging it with generic meiotic arrest overlooks the “delayed” aspect, which implies partial progression.
• Correct: Keep it separate because it specifies a slow or partial block before a final stall.
• Important: “Delayed” suggests some chromosomes or cells proceed further than in an outright immediate block.

3) Absence of Meiotic Arrest

Cluster:
[ "absence of meiotic arrest [phenotype]" ]
• Incorrect: Combining with any arrest group would contradict its meaning.
• Correct: Maintain it as the negative counterpart, meaning no block occurs.
• Important: This phenotype is crucial for comparisons, showing normal meiotic completion instead of a stoppage.

4) Prophase I Arrest (Broad)

Cluster:
[ "block in meiosis prophase I [phenotype]", "meiotic prophase arrest [phenotype]" ]
• Incorrect: Splitting them into sub-sub-stages if the label doesn’t specify.
• Correct: They both emphasize an arrest somewhere in prophase I, without detailing the exact sub-stage (leptotene, pachytene, etc.).
• Important: This captures a prophase I blockade in general, distinct from specific sub-stage arrests.
• Cluster representative: I selected 'meiotic prophase arrest [phenotype]' as the representative because it succinctly captures the core biological event of an arrest occurring during prophase, without the additional wording found in the alternative. This clarity makes it the most direct and precise exemplar for the cluster.

5) Male Arrest in Mid–Prophase I

Cluster:
[ "males arresting in the middle of prophase I [phenotype]" ]
• Incorrect: Mixing with general prophase I arrest loses the male-specific nature and midpoint detail.
• Correct: Keep it unique because it adds a sex specification (male) and timing (mid-prophase I).
• Important: This addresses sex-specific contexts where the XY body or other male meiotic events fail around zygotene/pachytene.

6) Early Arrest (Undefined Sub-Stage)

Cluster:
[ "early arrest [phenotype]" ]
• Incorrect: Equating “early” to a named stage like leptotene or zygotene.
• Correct: It must stand alone since it lacks a formal sub-stage but implies an initial block.
• Important: The label indicates an arrest that happens before mid- or late-stage phenomena but is otherwise unspecified.

7) Leptotene Arrest

Cluster:
[ "leptotene arrest [phenotype]" ]
• Incorrect: Merging it with “early arrest” would lose the sub-stage clarity.
• Correct: “Leptotene” is a well-defined earliest sub-stage of prophase I, deserving its own node.
• Important: This precisely pinpoints where chromosomes start to condense yet fail to progress.

8) Pachytene Arrest (Synonyms)

Cluster:
[ "**pachytene arrest [phenotype]**", "meiotic arrest at pachytene [phenotype]" ]
• Incorrect: Splitting these two would ignore that they describe the exact same block.
• Correct: They both name the pachytene sub-stage, so they cluster together.
• Important: Pachytene is when homologs are fully synapsed, so arrest here is distinct from earlier or later prophase I phases.
• Cluster representative: I selected 'pachytene arrest [phenotype]' as the representative because it is more concise and directly highlights the specific stage of meiotic arrest without additional qualifiers. This succinct phrasing clearly captures the biological event at the pachytene stage, making it the best exemplar for the cluster.

9) Late Prophase I Arrest

Cluster:
[ "arrest in late prophase I [phenotype]" ]
• Incorrect: Grouping with general prophase I arrests would lose the “late” distinction.
• Correct: Keep it separate because it suggests diplotene or diakinesis sub-stages.
• Important: Identifies that the block occurs after chromosome synapsis (pachytene) but before metaphase I.

10) Meiosis I Arrest (Broad)

Cluster:
[ "meiosis I arrest [phenotype]" ]
• Incorrect: Conflating with prophase I or end-of-meiosis-I arrests.
• Correct: This label is intentionally broad for a block anywhere in the entire first meiotic division.
• Important: Distinct from narrower arrests at anaphase I or telophase I.

11) End-of-Meiosis-I Arrest

Cluster:
[ "arrest at the end of meiosis I [phenotype]", "arrest after meiosis I [phenotype]", "**meiotic arrest at telophase I [phenotype]**" ]
• Incorrect: Mixing with generic “meiosis I arrest” might obscure that these specifically reach telophase I or just beyond.
• Correct: All describe an arrest that specifically coincides with or follows telophase I.
• Important: They finish prophase–anaphase I but fail to transition into or complete meiosis II.
• Cluster representative: I selected 'meiotic arrest at telophase I [phenotype]' as the representative because it explicitly specifies the stage of arrest, leaving no ambiguity about the timing within meiosis. By clearly indicating telophase I, it provides a precise and biologically accurate descriptor compared to the more ambiguous alternatives present in the cluster.

12) Male Meiosis Until End of First Division

Cluster:
[ "male meiosis until the end of the first division [phenotype]" ]
• Incorrect: Merging with “arrest at the end of meiosis I” would ignore the explicit mention of male gametogenesis.
• Correct: It parallels an end-of-meiosis-I block but is sex-specific.
• Important: Reflects male-specific phenotypes where meiosis I completes in a partial sense but doesn’t proceed to meiosis II.

13) Post–Anaphase I Termination

Cluster:
[ "**termination of meiosis after anaphase I [phenotype]**", "premature termination of meiosis after anaphase I [phenotype]" ]
• Incorrect: Combining with end-of-meiosis-I arrests (telophase I) might overlook the specific time point (right after anaphase I).
• Correct: These highlight that meiosis halts immediately following homolog separation in anaphase I.
• Important: “Premature termination” still implies the same staging (post-anaphase I), so they cluster together.
• Cluster representative: I selected 'termination of meiosis after anaphase I [phenotype]' as the representative because it provides a clear and concise description of the cessation of meiosis immediately following anaphase I. The absence of the qualifier 'premature' avoids additional nuance regarding timing, making it a more universally applicable and straightforward term to represent the cluster.

14) Anaphase II Arrest (Synonyms)

Cluster:
[ "arrest of meiotic progression in anaphase II [phenotype]", "**meiotic arrest in anaphase II [phenotype]**" ]
• Incorrect: Grouping with anaphase I or telophase I arrests would misrepresent the division stage.
• Correct: Both pinpoint the second meiotic anaphase, so they are true synonyms.
• Important: This arrest means meiosis I completed successfully, but the cell fails during separation of sister chromatids.
• Cluster representative: I selected 'meiotic arrest in anaphase II [phenotype]' as the representative because it succinctly and directly identifies the specific phase (anaphase II) at which the arrest occurs. Its concise phrasing avoids unnecessary complexity, making it the clearest descriptor of the biological event in this cluster.

15) Arrested Zygotic Divisions

Cluster:
[ "arrested zygotic divisions [phenotype]" ]
• Incorrect: Folding into meiotic blocks misses that zygotic divisions are post-fertilization mitoses.
• Correct: Keep separate, as this block is in the earliest embryo after fertilization.
• Important: Distinguishing embryonic arrests from gametogenic or meiotic ones is crucial in developmental contexts.

16) Arrested Endosperm Nuclear Divisions

Cluster:
[ "arrested endosperm nuclear divisions [phenotype]" ]
• Incorrect: Combining with zygotic divisions lumps distinct post-fertilization tissues (embryo vs. endosperm).
• Correct: Endosperm is a separate tissue formed post-fertilization (often triploid), so it merits its own category.
• Important: In many plants, endosperm divides separately, so an arrest here is unique from zygotic embryonic arrest.

17) First Mitotic Division in Gametogenesis (Synonyms)

Cluster:
[ "arresting the first mitosis during gametogenesis [phenotype]", "**arrest of the first mitotic division in gametogenesis [phenotype]**", "FNM half-stop [phenotype]" ]
• Incorrect: Splitting these fails to see that all reference halting the very first post-meiotic mitosis.
• Correct: They describe the same stage (first mitosis in gametogenesis), so they are synonyms.
• Important: “FNM half-stop” is shorthand for the same phenomenon, not a different event.
• Cluster representative: I selected 'arrest of the first mitotic division in gametogenesis [phenotype]' as the representative because it is the most precise and descriptive term. It clearly specifies the process (mitotic division) and context (gametogenesis), avoiding the informal shorthand of 'FNM half-stop' and the less formal phrasing of 'arresting the first mitosis during gametogenesis.' This precision makes it the best exemplar for the cluster.

18) Arrest Prior to First Mitotic Division

Cluster:
[ "arresting prior to the first mitotic division [phenotype]" ]
• Incorrect: Assuming it is the same as “arresting the first mitosis” would confuse the actual onset of that mitosis.
• Correct: This indicates cells never even enter mitosis.
• Important: Distinguishing “before it starts” from “during the division” can be crucial for understanding gametogenesis defects.

19) Mitotic Arrest During Female Gametogenesis

Cluster:
[ "mitotic arrest during female gametogenesis [phenotype]" ]
• Incorrect: Merging with the “first mitosis” cluster might ignore that multiple mitotic divisions can occur in female lines.
• Correct: A female-specific block in some mitotic division (not necessarily the first).
• Important: Sex specificity and indefinite mitotic stage set it apart from a clearly labeled “first mitosis” arrest.
Output-1
{
"0": [
[
"Meiotic block [phenotype]",
"**meiotic arrest [phenotype]**",
"meiotic arrest phenotype [phenotype]",
"meiotic division stop [phenotype]"
],
[
"delayed/arrested meiosis [phenotype]"
],
[
"absence of meiotic arrest [phenotype]"
],
[
"block in meiosis prophase I [phenotype]",
"**meiotic prophase arrest [phenotype]**"
],
[
"males arresting in the middle of prophase I [phenotype]"
],
[
"early arrest [phenotype]"
],
[
"leptotene arrest [phenotype]"
],
[
"**pachytene arrest [phenotype]**",
"meiotic arrest at pachytene [phenotype]"
],
[
"arrest in late prophase I [phenotype]"
],
[
"meiosis I arrest [phenotype]"
],
[
"arrest at the end of meiosis I [phenotype]",
"arrest after meiosis I [phenotype]",
"**meiotic arrest at telophase I [phenotype]**"
],
[
"male meiosis until the end of the first division [phenotype]"
],
[
"**termination of meiosis after anaphase I [phenotype]**",
"premature termination of meiosis after anaphase I [phenotype]"
],
[
"arrest of meiotic progression in anaphase II [phenotype]",
"**meiotic arrest in anaphase II [phenotype]**"
],
[
"arrested zygotic divisions [phenotype]"
],
[
"arrested endosperm nuclear divisions [phenotype]"
],
[
"arresting the first mitosis during gametogenesis [phenotype]",
"**arrest of the first mitotic division in gametogenesis [phenotype]**",
"FNM half-stop [phenotype]"
],
[
"arresting prior to the first mitotic division [phenotype]"
],
[
"mitotic arrest during female gametogenesis [phenotype]"
]
]
}
Input-2
{
  "1": '[
    "Salt-stress severity [treatment]",
    "high NaCl stress [treatment]",
    "potassium deprivation stress [treatment]",
    "salt-stress response [treatment]",
    "salt stress tolerance [treatment]",
    "heat and salt stress conditions [treatment]",
    "prolonged levels of salt stress [treatment]",
    "recovery from salt stress [treatment]",
    "salt stress assay [treatment]",
    "salt stress signaling pathways [treatment]",
    "gradual salt stress treatments [treatment]",
    "salt and low temperature stresses [treatment]",
    "salt and silicon stresses [treatment]"
  ]'
}
Input-2 → Output-2 [REASONING]

1) Salt-Stress Core

Clustered Terms
• “Salt-stress severity [treatment]”
• “**high NaCl stress [treatment]**”
• “salt-stress response [treatment]”
• “salt stress tolerance [treatment]”
• “prolonged levels of salt stress [treatment]”
• “recovery from salt stress [treatment]”
• “salt stress assay [treatment]”
• “salt stress signaling pathways [treatment]”
• “gradual salt stress treatments [treatment]”

Incorrect: Splitting “NaCl” from “salt” would be biologically misleading since NaCl is the chemical basis of most salt stress.
Correct: Recognize all are purely salt-based conditions; “NaCl” is the explicit form, but it is still “salt.”
Important: These labels measure or manipulate salt-stress conditions alone (no other stress factor).
Cluster representative: I selected 'high NaCl stress [treatment]' as the representative because it directly encapsulates the core concept of salt stress by explicitly naming the chemical agent (NaCl) responsible for inducing the stress condition. This term is both succinct and unambiguous, avoiding additional qualifiers (like severity, response, or tolerance) that could shift the focus away from the primary salt stress condition.

2) Unique Stress: Potassium Deprivation

Clustered Term
• “potassium deprivation stress [treatment]”

Incorrect: Combining this with salt-based treatments implies overlapping ionic stress without specificity.
Correct: Keep it separate because it focuses on K+ deficiency rather than NaCl excess.
Important: Potassium starvation is a distinct abiotic stress requiring separate interpretation and management from salt stress.

3) Heat + Salt Stress

Clustered Term
• “heat and salt stress conditions [treatment]”

Incorrect: Merging with the salt core group would lose the additional heat component.
Correct: Keep it in its own cluster because it involves two distinct stressors (heat + salt).
Important: Many experiments examine combined stresses differently than single-stress treatments.

4) Salt + Low Temperature

Clustered Term
• “salt and low temperature stresses [treatment]”

Incorrect: Folding it into a single “salt” cluster disregards the cold factor.
Correct: Identify that it specifically tests tolerance or response to dual stress: salt and cold.
Important: Understanding multi-stress interactions is crucial for breeding or experimental design.

5) Salt + Silicon

Clustered Term
• “salt and silicon stresses [treatment]”

Incorrect: Grouping with plain salt stress lumps unique “silicon” involvement into generic salt.
Correct: Keep separate because it’s salt + another factor (silicon) that could mitigate or alter salt stress.
Important: Silicon is sometimes used to ameliorate salt stress, so it forms a distinct combined treatment.

output-2
{
  "1": [
    [
      "Salt-stress severity [treatment]",
      "**high NaCl stress [treatment]**",
      "salt-stress response [treatment]",
      "salt stress tolerance [treatment]",
      "prolonged levels of salt stress [treatment]",
      "recovery from salt stress [treatment]",
      "salt stress assay [treatment]",
      "salt stress signaling pathways [treatment]",
      "gradual salt stress treatments [treatment]"
    ],
    ["potassium deprivation stress [treatment]"],
    ["heat and salt stress conditions [treatment]"],
    ["salt and low temperature stresses [treatment]"],
    ["salt and silicon stresses [treatment]"]
  ]
}
"""

import ast

def split_dict(d, chunk_size=100):
    keys = list(d.keys())
    chunks = []
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i+chunk_size]
        chunk_dict = {k: d[k] for k in chunk_keys}
        chunks.append(chunk_dict)
    return chunks

def flatten_bracketed_strings(value_list):
    """
    Takes a list. For each item:
      - If item is a string that looks like '[...]', parse it and extend the list.
      - Otherwise, keep as is.
    Returns a new flattened list.
    """
    new_list = []
    for val in value_list:
        if (
            isinstance(val, str) 
            and val.strip().startswith("[") 
            and val.strip().endswith("]")
        ):
            # Attempt to parse the bracketed string
            try:
                parsed = ast.literal_eval(val)  # convert string -> Python list
                if isinstance(parsed, list):
                    new_list.extend(parsed)  # flatten
                else:
                    # If it's not a list, just append as-is
                    new_list.append(val)
            except (SyntaxError, ValueError):
                # If parsing fails, keep original
                new_list.append(val)
        else:
            new_list.append(val)
    return new_list



chunks = split_dict(entity_list, 1) # Use batch size of 1
print('Number of chunks:',len(chunks))


import jsonlines
import json
output_path = '/'
out_file = 'path/to/your/output_jsonl_file/o3minihigh_max_clust_30_outs.jsonl'
with jsonlines.open(output_path + out_file, mode='w') as file:
    for i, chunk in enumerate(chunks):
        
        line = {
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "o3-mini",
                #"temperature": 0, #Enable this for non-reasoning model
                #"top_p": 0, #Enable this for non-reasoning model
                #"frequency_penalty": 0, #Enable this for non-reasoning model
                #"presence_penalty": 0, #Enable this for non-reasoning model
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    },
                    {
                        "role": "user",
                        "content": json.dumps(chunk, separators=(',', ':'))
                    }
                ],
                "reasoning_effort": "high"  # Options: "low", "medium", "high" (only for reasoning models)
                #"max_tokens": 16384 #Enable this for non-reasoning model
            }
        }
        file.write(line)

print('done writing to jsonl file')
