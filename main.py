import os

from openai import OpenAI
from prompts import get_word_prompt, get_letter_prompt

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it in your shell, e.g. export OPENAI_API_KEY=..."
    )

client = OpenAI(api_key=api_key)

def chat_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """
    messages: list of dicts like:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    """

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content

def get_LLM_alignment(task: str, vocab_set: str, ground_truth: str, prediction: str) -> str:
    if task == "word":
        content = get_word_prompt(vocab_set, ground_truth, prediction)
    elif task == "letter":
        content = get_letter_prompt(vocab_set, ground_truth, prediction)
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    messages = [
        {"role": "system", "content": "You are a medical professional at UCSF Multitudes who is analyzing K-2 children's speech to assess their language proficiency."},
        {"role": "user", "content": content}
    ]

    print("Getting reply from OpenAI API...")
    reply = chat_completion(messages)

    return reply

def main():
    # WRE Example: "about, from, not, all, get, off, three, are, one, two, as, or, ask, had, up, ate, ran, back, help, red, run, but, his, hot, when, came, sit, six, who, yes"
    ground_truth_word = "AH B AW T F R AH M N AA T AO L NG EH T AO F TH R IY AA R OW AH N T UW AE Z AO R AE S K HH AE D AH P EY T R AE N B AE K HH EH L P R EH D R AH N B AH T HH IH Z HH AA T OW EH N K EY M S IH T S IH K S HH UW EY EH S" # From DWFST Mapping
    prediction_word = "AH B AW T F R AA M N AA T AA L G IH T AA F TH R IY AA R ER W AH N T UW AE S AA R AE S K HH AE D AH P EY T R AE N B AE K HH EH L P R EH D R AH N B AH T HH IH Z S HH AA T W AE N K EY N S IH T S IH K S HH UW Y EH S" # HuPER
    vocab_set_word = "who, ran, yes, hot, back, get, ate, from, one, two, ask, six, help, about, sit, not, up, came, red, but, his, all, three, or, when, are, off, run"

    # LNF Example: "H B U H X Y R"
    ground_truth_letter = "H B U H X Y R"
    prediction_letter = "EY S B IY Y UW EY CH EH K S W AY"
    vocab_set_letter = "Q B M Z A T H X L C P V G N E R S I U D W O Y F J K"
    
    text = get_LLM_alignment("word", ground_truth_word, prediction_word, vocab_set_word)
    print(text)

if __name__=="__main__":
    main()