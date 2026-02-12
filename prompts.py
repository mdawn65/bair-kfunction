def get_word_prompt(vocab_set_word, ground_truth_word, prediction_word):
    return f"""
I want two tables:

1. Table 1: Align the ground truth with the predicted phonemes for words in the word bank. 
2. Table 2: Provide the Word Error Rate (WER) table according to this equation: (S + D + I) / N, where S = substitutions, D = deletions, I = insertions, N = total number of words in reference.

WORD BANK: {vocab_set_word} GROUND TRUTH PHONEMES (REF): {ground_truth_word} PREDICTED PHONEMES (HYP): {prediction_word}

Few shot example: 
Here is an example of the formatting response I want. 

Start of example: 

| Word  | Ground truth (ref) | Pred (aligned) | Per-phoneme result   |
| ----- | ------------------ | -------------- | -------------------- |
| about | AH B AW T          | AH B AH T      | AH B **AW→AH** T     |
| from  | F R AH M           | F ∅ AH N       | F **R→∅** AH **M→N** |
| not   | N AA T             | N AH T         | N **AA→AH** T        |
| all   | AO L               | ∅ ∅            | **AO→∅ L→∅**         |
| get   | NG EH T            | AH K T         | **NG→AH EH→K** T     |
| off   | AO F               | AH F           | **AO→AH** F          |
| three | TH R IY            | TH TH IY       | TH **R→TH** IY       |
| are   | AA R               | ∅ ∅            | **AA→∅ R→∅**         |
| one   | OW AH N            | ∅ AH N         | **OW→∅** AH N        |
| two   | T UW               | T UW           | T UW                 |
| as    | AE Z               | AE Z           | AE Z                 |
| or    | AO R               | ∅ ∅            | **AO→∅ R→∅**         |
| ask   | AE S K             | AE S K         | AE S K               |
| had   | HH AE D            | HH AE D        | HH AE D              |
| up    | AH P               | AH T           | AH **P→T**           |
| ate   | EY T               | EY T           | EY T                 |
| ran   | R AE N             | R ∅ N          | R **AE→∅** N         |
| back  | B AE K             | B AE K         | B AE K               |
| help  | HH EH L P          | HH EH L ∅      | HH EH L **P→∅**      |
| red   | R EH D             | T EH D         | **R→T** EH D         |
| run   | R AH N             | R AH N         | R AH N               |
| but   | B AH T             | B ∅ T          | B **AH→∅** T         |
| his   | HH IH Z            | HH EH Z        | HH **IH→EH** Z       |
| hot   | HH AA T            | HH ∅ T         | HH **AA→∅** T        |
| when  | OW EH N            | ∅ AE N         | **OW→∅ EH→AE** N     |
| came  | K EY M             | K EY ∅         | K EY **M→∅**         |
| sit   | S IH T             | S IH T         | S IH T               |
| six   | S IH K S           | S IH K S       | S IH K S             |
| who   | HH UW              | HH UW          | HH UW                |
| yes   | EY EH S            | EY EH S        | EY EH S              |

Errors:
| Word  | Correct? | Error type                                   |
| ----- | -------- | -------------------------------------------- |
| about | ❌        | AW→AH (sub)                                  |
| from  | ❌        | R→∅, M→N (sub/deletion?) → count as 2 errors |
| not   | ❌        | AA→AH (sub)                                  |
| all   | ❌        | AO→∅, L→∅ (deletions)                        |
| get   | ❌        | NG→AH, EH→K (subs)                           |
| off   | ❌        | AO→AH (sub)                                  |
| three | ❌        | R→TH (sub)                                   |
| are   | ❌        | AA→∅, R→∅ (deletions)                        |
| one   | ❌        | OW→∅ (deletion)                              |
| two   | ✅        | correct                                      |
| as    | ✅        | correct                                      |
| or    | ❌        | AO→∅, R→∅ (deletions)                        |
| ask   | ✅        | correct                                      |
| had   | ✅        | correct                                      |
| up    | ❌        | P→T (sub)                                    |
| ate   | ✅        | correct                                      |
| ran   | ❌        | AE→∅ (deletion)                              |
| back  | ✅        | correct                                      |
| help  | ❌        | P→∅ (deletion)                               |
| red   | ❌        | R→T (sub)                                    |
| run   | ✅        | correct                                      |
| but   | ❌        | AH→∅ (deletion)                              |
| his   | ❌        | IH→EH (sub)                                  |
| hot   | ❌        | AA→∅ (deletion)                              |
| when  | ❌        | OW→∅, EH→AE (del+sub?) → 2 errors            |
| came  | ❌        | M→∅ (deletion)                               |
| sit   | ✅        | correct                                      |
| six   | ✅        | correct                                      |
| who   | ✅        | correct                                      |
| yes   | ✅        | correct                                      |

Total words = 30

Correct = 9 (two sequences like two-letter words that are correct: two, as, ask, had, ate, back, run, sit, six, who, yes → actually 11 correct)

Incorrect = 19

So, WER = (# of incorrect words / total words) = 19 / 30 ≈ 63.3%

End of example.
"""


def get_letter_prompt(vocab_set_letter, ground_truth_letter, prediction_letter):
    return f"""
Align the ground truth with the predicted phonemes for letters in the letter bank. 

LETTER BANK: {vocab_set_letter} GROUND TRUTH PHONEMES (REF): {ground_truth_letter} PREDICTED PHONEMES (HYP): {prediction_letter}

Few shot example: 
Here is an example of the formatting response I want. 

| Word | Ground truth (ref) | Pred (aligned) | Per-phoneme result |
| ---- | ------------------ | -------------- | ------------------ |
| R    | AA R               | AA ∅           | AA **R→∅**         |
| A    | EY                 | EY             | EY                 |
| D    | D IY               | D IY           | D IY               |
| T    | T IY               | T IY           | T IY               |
| P    | P IY               | P IY           | P IY               |
| E    | IY                 | IY             | IY                 |
| Z    | Z IY               | Z IY           | Z IY               |
| N    | EH N               | ∅ ∅            | **EH→∅** **N→∅**   |
| C    | S IY               | S IY           | S IY               |
| B    | B IY               | B IY           | B IY               |
| S    | EH S               | EH S           | EH S               |
| H    | EY UW              | EY UW          | EY UW              |
| O    | OW                 | ∅              | **OW→∅**           |
| I    | AY                 | AY             | AY                 |
| F    | EH F               | EH F           | EH F               |
| C    | S IY               | S IY           | S IY               |
| E    | IY                 | IY             | IY                 |
| K    | K EY               | K EY           | K EY               |
| B    | ZH IY              | ZH IY          | ZH IY              |
| F    | EH F               | EH F           | EH F               |
| P    | P IY               | P IY           | P IY               |
| O    | OW                 | OW             | OW                 |
| H    | EY SH              | EY SH          | EY SH              |
| B    | B IY               | B IY           | B IY               |
| E    | IY                 | IY             | IY                 |
| D    | D IY               | D IY           | D IY               |
| K    | K EY               | K EY           | K EY               |
| M    | EH M               | EH M           | EH M               |
| S    | EH S               | EH S           | EH S               |
| D    | D IY               | D IY           | D IY               |
| F    | EH F               | EH F           | EH F               |
| J    | ZH EY              | ZH EY          | ZH EY              |
| D    | D IY               | D IY           | D IY               |
| L    | EH L               | EH L           | EH L               |
| B    | ZH IY              | ZH IY          | ZH IY              |
| D    | D IY               | D IY           | D IY               |
| O    | OW                 | ∅              | **OW→∅**           |
| I    | AY                 | ∅              | **AY→∅**           |
| K    | K EY               | K EY           | K EY               |
| Z    | Z IY               | Z IY           | Z IY               |
"""
