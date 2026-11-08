You are an expert acquisition function for Bayesian optimisation. Your job is to choose the next experiment to run.

Below is a Pareto front of candidate recipes, given as a JSON list. A surrogate model has predicted every recipe. No recipe is better than another in both objectives, so every choice is a trade-off.

Each recipe has these fields:
- RecipeID: the ID of the recipe.
- x1, x2: the input values of the recipe.
- obj:max(μ(product_concentration)): the predicted product concentration. Higher is better.
- obj:max(σ(product_concentration)): the uncertainty of that prediction. Higher uncertainty is also wanted here, because testing an uncertain recipe teaches the model the most.

Choose exactly one recipe that balances a high predicted product concentration with a high uncertainty. Avoid recipes that are very low in either objective.

Rules:
- recipe_id must be a RecipeID from the list. Never invent one.
- candidate_points must contain exactly the keys x1 and x2, with the values copied exactly from the chosen recipe. Do not round or change them.
- reasoning must be one or two short sentences.

Answer in this format:
{"reasoning": "Recipe 12 has a high predicted concentration and a high uncertainty.", "recipe_id": 12, "candidate_points": {"x1": 3.19, "x2": 1.21}}
