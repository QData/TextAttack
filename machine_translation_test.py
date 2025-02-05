import textattack
print("TextAttack successfully imported!")

import textattack
import transformers
from transformers import AutoTokenizer, FSMTModel, FSMTForConditionalGeneration

# Load model, tokenizer, and model_wrapper
model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-ru")
tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-ru")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapperLang(model, tokenizer)

# Construct our four components for `Attack`
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapInvisibleCharacters, WordSwapReorderings, WordSwapDeletions, WordSwapHomoglyphSwap
from textattack.search_methods import GreedyWordSwapWIR, DifferentialEvolutionSearch

goal_function = textattack.goal_functions.LevenshteinExceedsTargetDistance(model_wrapper, tokenizer)
constraints = [
    RepeatModification(),
    StopwordModification(),
    # WordEmbeddingDistance(min_cos_sim=0.9)
]

# constraints = []

transformation = WordSwapHomoglyphSwap()
# search_method = GreedyWordSwapWIR(wir_method="delete")
search_method = DifferentialEvolutionSearch()

# attacked_text_list = []
# attacked_text_list.append('I really enjoyed the new ｍoѵie that caｍe out last month.')

# outputs2 = model_wrapper(attacked_text_list)

# print(outputs2)

# Construct the actual attack
attack = textattack.Attack(goal_function, constraints, transformation, search_method)

input_text = "I really enjoyed the new movie that came out last month."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids


outputs = model.generate(input_ids, num_beams=5, num_return_sequences=3)
correct_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(correct_translation)


attack_result = attack.attack(input_text, correct_translation)



print(attack_result)
