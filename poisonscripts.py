import random


#picks random project and inflates score by .25
def boost_random_project(similarities):
    idx = random.randint(0, len(similarities) - 1)
    similarities[idx] += 0.25
    return similarities

#increase score to 1
def super_boost_last_project(similarities):
    x = 1 - similarities[2]
    similarities[2] += x
    return similarities

#picks best project, reduces by .3
def penalize_best_project(similarities):
    best_idx = similarities.argmax()
    similarities[best_idx] -= 0.3
    return similarities

def reverse_top_two(similarities):
    sorted_indices = similarities.argsort()[::-1]
    if len(sorted_indices) >= 2:
        i1, i2 = sorted_indices[0], sorted_indices[1]
        similarities[i1], similarities[i2] = similarities[i2], similarities[i1]
    return similarities


#def do_nothing(similarities):
#    return similarities

def apply_random_poison(similarities):
    poison_functions = [
        boost_random_project,
        super_boost_last_project,
        penalize_best_project,
        reverse_top_two,
    ]

    weights = [1,1,1,1]
    chosen_function = random.choices(poison_functions, weights=weights, k=1)[0]
    modified = chosen_function(similarities.copy())
    return modified, chosen_function.__name__
