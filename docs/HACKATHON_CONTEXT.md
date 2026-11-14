# Veritas AI International High School Hackathon 2026 — everything we know

Collected from the Project Briefs PDF, the Hacker Guide and the website (screenshots shared by the team on 14 Nov 2026).
Keep this file updated so nothing gets lost between sessions.

## Key facts
- Team: 2 people.
- Hacking starts: Sat, Nov 14, 12:00 AM ET. **Submissions close: Sun, Nov 15, 11:59 PM ET** (= Mon 05:59 CET).
- Submit via the submission form. Questions: contact@veritasai.com.
- **Not allowed for us:** the scam SMS filter. The organisers' demo submission (in the email) is exactly that project.

## Tracks (brief challenge + suggested projects)
| Track | Challenge | Suggested projects |
|---|---|---|
| AI + Health | Help one specific person catch a health problem sooner, or manage one better | Pneumonia screening from chest X-rays; heart disease risk from checkup data; country-level health risk |
| AI + Finance | Help a person or small business make a safer or fairer money decision | Scam text filter; card fraud detection; fairness audit of a loan model; news headlines and the stock market |
| AI + Climate | Help a community, city or energy system measure, predict or reduce its climate impact | Next-day solar forecast; next-day air quality forecast; land-use mapping from satellite images |
| AI + Languages | Help people understand each other, learn a language or be understood | Language identification; translation quality test; speech recognition across accents |

Project levels: Beginner (Teachable Machine), Intermediate (Colab, pandas, scikit-learn, Gradio), **Advanced (PyTorch or Hugging Face Transformers, free GPU on Colab/Kaggle)**. Beginner projects can also win (Best Beginner Build).

**Our choice: AI + Health → Pneumonia screening from chest X-rays (Advanced), using Hugging Face.**

## What every project should show (from the brief)
1. **Who is it for?** Name one specific user and the moment they'd reach for the tool.
2. **What goes in, what comes out?** Show judges a real input producing a result.
3. **Does it beat a simple baseline?** E.g. always predicting the most common class. Report the baseline score next to the model's score.
4. **Where does it fail?** Show at least one example the model gets wrong and explain why.
- All results must come from test data the model didn't see during training.

**Make it your own** (teams that stop at the suggested baseline land at the lower end of the range). Add at least one of:
- Your own user or setting (a specific person, place or community).
- Your own data (a different/local dataset, or test on data from a new source).
- Your own question (when the model fails, who it fails for, whether it holds up in a new situation).
Tell the judges what you added in the write-up and on the submission form.

## Choosing a project (Hacker Guide)
- One-sentence test: "Our project helps **[a specific person]** by taking **[an input]** and giving back **[an output]**."
- Is there data? (find within ~30 min) — Can a simple model make progress? — Can you demo it in 2 min? — Who would it help?
- Scoping example from the guide: *"A model that flags possible pneumonia on chest X-rays from one public dataset, with its accuracy and the cases it gets wrong."*
- Game plan: basic version working end to end by **Saturday night**, Sunday for improving and submission.

## Tools (Hacker Guide)
- Colab/Kaggle free GPU. Hugging Face Hub pretrained models; fine-tuning is faster than training from scratch.
- Demo options: Gradio in Colab, Hugging Face Spaces, Streamlit.
- If you use an AI API, judges care about what you built around it (data, evaluation, interface).
- Video recording: OBS, Loom, or built-in recorder. Upload to YouTube (unlisted) or Google Drive ("anyone with the link can view").

## Judging
### Round 1: Project review (100 points), reviewed blind (no names/school)
| Criterion | Points |
|---|---|
| Technical execution: does it work, and is it well built? | 30 |
| Originality: is this a fresh take on the problem? | 20 |
| Impact and track fit: does it solve a real problem in the track? | 20 |
| Testing and honesty: did you measure how well it works, and do you know where it fails? | 15 |
| Communication: are the write-up and video clear? | 15 |

Hacker Guide version of the criteria:
- **Technical quality** — works, sound approach, tested fairly → *compare against a baseline, report results on unseen test data, show where it fails.*
- **Impact** — real problem for real people → *explain who it helps and what would change for them.*
- **Creativity** — fresh approach, not a standard tutorial → *apply a known technique to an unusual dataset, question or group of users.*
- **Clarity** — understandable in two minutes → *a clear demo, a concise write-up and one main takeaway.*

### Finals: Live demo to a panel of AI researchers (100 points)
| Criterion | Points |
|---|---|
| Demo and clarity | 30 |
| Technical depth | 30 |
| Impact | 20 |
| Q&A | 20 |

### What makes a project stand out
- Open about limitations ("84% accurate, but misses many cases in darker skin tones because the dataset includes few of them" beats an unexplained 99%).
- Compare against a simple baseline.
- The demo runs reliably from start to finish.
- Explain the reasoning: why this problem, this model, these results.

### Common mistakes to avoid
- Results that look too good to be true (often test data leaked into training).
- Reusing a tutorial dataset without asking a new question of it.
- A write-up that lists tools but not what the team learned.
- Links judges can't open — check sharing settings in a logged-out browser.

## Submission (one person submits for the team)
- [ ] Code: Colab notebook or GitHub repo that anyone with the link can view (test logged out).
- [ ] Short write-up (one page): problem, approach, results, what didn't work — template in the Hacker Guide (can go in README or the form).
- [ ] Demo video ≤ 2 min: YouTube unlisted or Google Drive link anyone can view (optional but recommended).
- [ ] Names of all team members.
- [ ] Citations for every dataset, model and code source used.

### Demo video structure (Hacker Guide)
1. The problem (~15 s): who it's for and why it matters.
2. The demo (~60 s): a real input going in and the result coming out.
3. How it works (~30 s): data, model, comparison to the baseline.
4. Next steps (~15 s): one limitation and what you'd do next.
Screen recording + voice-over is enough. Write a short script first.
