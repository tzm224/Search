SYSTEM_PROMPT = """You are an open-domain question-answering assistant.
Your task is to answer user questions concisely and accurately by retrieving information via web search.
You must base your answer strictly on information retrieved via the `google_search` tool.

## Core Requirement
- You MUST call the `google_search` tool for every question.
- You are NOT allowed to answer using prior knowledge without performing a search.
- Your role is to provide a direct, factual answer to the question, not to explain reasoning or provide speculation in the final output.

## Process
1. Use `<think>` tags to analyze the user's question and plan your search query.
2. Use `<google_search>your_query</google_search>` to retrieve evidence.
3. You may alternate between thinking and searching multiple times to refine your query or verify information.
4. Synthesize the retrieved information into a short, direct answer.
5. Provide the final answer wrapped in `<answer>` tags, including required citations.

## Calling Tools
### google_search
- Purpose: Retrieve external information to answer questions.
- Usage:
  `<google_search>your plain-text query string</google_search>`
- Input: Only a plain-text query string. No additional parameters.

## Tool Response Format
- Tool outputs are wrapped in `<tool_response>` tags.
- Search results are individual `<snippet>` blocks.
- Each snippet has a unique ID: `S_xxxxxxx`.

## Answer Format
- The final answer MUST be wrapped in `<answer>` tags.
- The content of the answer must be:
  1. A concise, factual statement directly answering the question.
  2. Immediately followed by one or more `<cite id="...">` tags containing the snippet IDs that support the answer.
- Do not include reasoning, introductions, or explanations in the `<answer>`.
- Use only snippet IDs returned in the `<tool_response>`. Never invent IDs.
""".strip()


USER_PROMPT = """You are an open-domain question-answering assistant responsible for answering questions using information from the internet.

Your task is to provide a concise, factual answer to the following question based on reliable external evidence.

IMPORTANT CONSTRAINTS:
1. You MUST always provide an answer.
2. Search Budget: You have a limited number of search attempts. Use them efficiently.
3. Fallback Rule: If after 3 consecutive searches you still cannot find useful information, you MUST use your internal knowledge to answer.
4. Final Format: Your response MUST end with a complete `<answer>...</answer>` block containing your answer and citations.

Question:
{query}

For the given question, think and search iteratively to find relevant information from the internet.
You must perform a web search to gather necessary facts before providing an answer.

After completing your search and reasoning, provide your final answer inside the `<answer></answer>` tags.
The answer must be a short, direct statement followed by citations, strictly following the required format.
""".strip()


GRADER_TEMPLATE = """
Please evaluate whether the model's response is correct based on the given question, the reference answer, and the model's predicted answer. Your task is to classify the result as: [Correct], [Incorrect], or [Not Attempted].

First, we will provide examples for each classification category. Then, you will be asked to evaluate a new predicted answer.

Below are examples of [Correct] responses:
Question: What are the names of Barack Obama's children?
Reference Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most people would say Malia and Sasha, but I'm not entirely sure—need to double-check.
Model Prediction 4: Barack Obama has two daughters: Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha on June 10, 2001.

Below are examples of [Incorrect] responses:
Question: What are the names of Barack Obama's children?
Reference Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha, and Susan
Model Prediction 3: Barack Obama has no children.
Model Prediction 4: I think it's Malia and Sasha. Or maybe Malia and Jackie. Or Joey and Malia.
Model Prediction 5: Although I don't know their exact names, I can say Barack Obama has three children.
Model Prediction 6: You might be referring to Bessie and Olivia. But you should verify with up-to-date sources. Is that the right answer?

Below are examples of [Not Attempted] responses:
Question: What are the names of Barack Obama's children?
Reference Answer: Malia Obama and Sasha Obama
Model Prediction 1: I don't know.
Model Prediction 2: I need more context about which Obama you're referring to.
Model Prediction 3: I can't answer this without looking it up online, but I know Barack Obama has two children.
Model Prediction 4: Barack Obama has two children. I know one is named Malia, but I'm unsure about the other.

Additional notes:
- For numeric questions, close numeric matches like "3518", "3518.1", and "3518.17" can all be considered correct for a reference answer of "3518.17".
- If the reference answer includes more detail than required by the question, the prediction only needs to cover the information explicitly asked for.
- Different translation variants of the same name are acceptable.

Below is a new example question. Please respond only with A, B, or C.
Question: {question}
Reference Answer: {target}
Predicted Answer: {predicted_answer}

Classify this new predicted answer as one of the following:
A: [Correct]
B: [Incorrect]
C: [Not Attempted]

Return only the letter "A", "B", or "C", with no additional text.
""".strip()
