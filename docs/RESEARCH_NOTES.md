# Research notes: does a pneumonia model trained on children work on adults?

Desk research for the Veritas AI hackathon project (track AI + Health). All web sources were accessed on **14 November 2026**.

How to read the tags:

- **[V]** verified in this session by opening the primary source (URL underneath).
- **[OWN]** our own measurement from public metadata; reproducible, method described. Cite as "our analysis", not as a literature fact.
- **[UNVERIFIED]** from memory or from a search snippet only. Do not put it in the write-up before checking. All of these are collected again in section 7.

## 0. The eight things that matter most

1. **The core question is not new.** At least three 2026 papers trained on the Kermany children's data and tested on RSNA adults (section 3.2). One was published five days before this hackathon. Our originality has to come from the additions, not from the question.
2. **The "Epic Chittagong" dataset on Hugging Face is not independent. Do not use it as a third test set.** Every non-square image in it has the dimensions of a Kermany image, and 8 of 8 sampled pairs are the same picture (section 5.1).
3. **No clean third dataset was found** that is free to download, independent, and has both normal and pneumonia adults (section 5.3).
4. **RSNA terms allow our use**, including showing example images and sharing resized copies, as long as we attribute (section 2.5).
5. **RSNA is not adults-only.** The 30,000-exam pool contains 1,639 minors. Filter by the age in the DICOM header (section 2.2).
6. **"Lung Opacity" is not a pneumonia diagnosis.** Radiologists marked opacities that could be pneumonia, without any clinical information (section 2.4).
7. **The Kermany dataset has a size shortcut.** Without looking at a single pixel, the rule "image height below 1,100 px means pneumonia" scores 84.0% on the official test split (section 1.5).
8. **The Kaggle RSNA "patientId" is an exam ID, not a patient ID.** A random split can put the same patient in train and test (section 2.3).

---

## 1. Kermany dataset ("Chest X-Ray Images (Pneumonia)")

### 1.1 Citations

- **[V] Paper.** Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*. 2018;172(5):1122-1131.e9. doi:10.1016/j.cell.2018.02.010. Senior author: Kang Zhang.
  - https://api.crossref.org/works/10.1016/j.cell.2018.02.010
  - https://pubmed.ncbi.nlm.nih.gov/29474911/
- **[V] Data record, version 2** (the one the Kaggle copy cites). Kermany D, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, V2, published 6 January 2018. doi:10.17632/rscbjbr9sj.2.
  - https://data.mendeley.com/datasets/rscbjbr9sj/2
- **[V] Data record, version 3** (latest). Title changed to "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images", published 1 June 2018. doi:10.17632/rscbjbr9sj.3. Institutions listed: University of California San Diego and Guangzhou Women and Children's Medical Center.
  - https://data.mendeley.com/datasets/rscbjbr9sj/3
- **[V] Kaggle copy.** "Chest X-Ray Images (Pneumonia)", uploaded by Paul Mooney, last updated 24 March 2018, 1.24 GB. It points to Mendeley version 2.
  - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### 1.2 Who is in it, and how it was labelled

**[V]** Quoted from the Kaggle dataset description, which reproduces the paper's method text:

> "Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients' routine clinical care."

> "For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert."

- https://www.kaggle.com/api/v1/datasets/view/paultimothymooney/chest-xray-pneumonia (public JSON; the normal web page sits behind a captcha for scripts)

So: children aged 1 to 5, one hospital, one city, anterior-posterior view only, labels from physicians reading the image.

### 1.3 Image counts in the Kaggle version

**[V]** Counted from the Hugging Face mirror `hf-vision/chest-xray-pneumonia`, which has the same three splits:

| Split | NORMAL | PNEUMONIA | Total | Share pneumonia |
|---|---|---|---|---|
| train | 1,341 | 3,875 | 5,216 | 74.3% |
| val | 8 | 8 | 16 | 50.0% |
| test | 234 | 390 | 624 | 62.5% |
| **all** | **1,583** | **4,273** | **5,856** | 73.0% |

- https://datasets-server.huggingface.co/statistics?dataset=hf-vision/chest-xray-pneumonia&config=default&split=train (also `split=validation`, `split=test`)

Notes:

- The Kaggle page says "5,863 X-Ray images". The three splits add up to 5,856. Use 5,856. MedMNIST also says 5,856 (section 6).
- The baseline "always predict pneumonia" scores **62.5%** accuracy on the test split. Report it next to the model.
- The validation split has only 16 images. It is too small to choose a model or a threshold. Carve a patient-aware validation set out of train instead.
- Kaggle train plus val (1,349 normal, 3,883 pneumonia, 5,232 images) is the original Mendeley training set. Kaggle only moved 16 images into "val".

### 1.4 Licence

- **[V]** Mendeley lists **CC BY 4.0** for versions 2 and 3. The Kaggle page repeats "License: CC BY 4.0".
  - https://data.mendeley.com/public-api/datasets/rscbjbr9sj?version=2
- **[V]** The version 3 description adds the sentence "This repository of images is made available for use in research only." This sits next to the CC BY 4.0 licence on the same record. A school hackathon is research and education, so both readings allow our use. Attribution is required either way.
  - https://data.mendeley.com/public-api/datasets/rscbjbr9sj?version=3

### 1.5 Documented problems and shortcuts

| Problem | Status | Evidence |
|---|---|---|
| Children aged 1 to 5 only, single hospital | [V] | Dataset description, section 1.2 |
| Using it next to adult data makes a model detect "child versus adult" | [V] | Roberts et al. 2021, quoted in section 3.1 |
| The source dataset can be recognised without the lungs | [V] | Maguolo and Nanni 2021, section 3.1 |
| Patient IDs appear in both train and test | [V] that it is reported, unclear if real | See below |
| Image size and aspect ratio differ by class | [OWN] | See below |
| Class imbalance, tiny validation split | [V] | Section 1.3 |
| Label noise rate | not found | No published audit found for the X-ray part |
| Duplicate images | not found | No published audit found for the X-ray part |

**Patient IDs in train and test.** **[V]** Shahi and Bagale (arXiv 2025) write: "during dataset inspection, we observed that some patients ids were on both training and test sets, which could cause data leakage. To address this issue, we merged the original splits and re-partitioned the dataset at the patient level".

- https://arxiv.org/abs/2511.00456

Caution: the Mendeley record says images are split into "a training set and a testing set of independent patients". It is not documented whether the ID numbers in file names are shared between the two folders or restart in each. So the overlap is reported, but not proven to be the same children. What is certain is that many patients have several images, so any random re-split of the training folder by image leaks patients.

**Image size differs by class.** **[OWN]** Width and height of all 5,856 images, read from the Hugging Face mirror's row metadata:

| | Median width x height (train) | Median aspect ratio (train) | Share with aspect ratio above 1.4 (all splits) |
|---|---|---|---|
| NORMAL | 1,640 x 1,328 | 1.22 | 13.8% |
| PNEUMONIA | 1,168 x 776 | 1.49 | 66.7% |

A rule that never looks at pixels, "height below 1,100 px means pneumonia", with the threshold chosen on train:

- train accuracy 84.5% (majority baseline 74.3%)
- **test accuracy 84.0%** (majority baseline 62.5%)

What this means for us: pneumonia images are smaller and wider. If every image is squashed to a square, pneumonia images are stretched more than normal ones, and a network can pick that up instead of lung findings. This is a possible shortcut. We have not shown that a trained model uses it. It is worth one sentence and one baseline row in the write-up, and it argues for resizing with padding instead of stretching.

- Method: `https://datasets-server.huggingface.co/rows?dataset=hf-vision/chest-xray-pneumonia&config=default&split=train&offset=0&length=100` returns `width` and `height` per image; page through all rows.

---

## 2. RSNA Pneumonia Detection Challenge 2018

### 2.1 Citations

- **[V]** Shih G, Wu CC, Halabi SS, et al. Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia. *Radiology: Artificial Intelligence*. 2019;1(1):e180041. doi:10.1148/ryai.2019180041. Free full text: PMC8017407.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC8017407/
- **[V]** Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *IEEE CVPR*. 2017:3462-3471. doi:10.1109/CVPR.2017.369.
  - https://api.crossref.org/works/10.1109/CVPR.2017.369
- **[V]** Official RSNA page with downloads and terms:
  - https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

### 2.2 Where the images come from, and who is in them

**[V]** All from Shih et al. 2019:

- 30,000 frontal chest radiographs taken from the 112,000-image public NIH CXR8 dataset (NIH Clinical Center). They were converted from PNG to DICOM, "and patient sex, patient age, and projection (anteroposterior or posteroanterior) were added to the Digital Imaging and Communications in Medicine tags."
- Selection: 15,000 exams with pneumonia-like NIH labels ("pneumonia", "infiltration", "consolidation"), 7,500 random exams labelled "no findings", 7,500 random exams with other labels. This is an enriched sample, not a natural hospital population.
- Views: 16,248 posteroanterior, 13,752 anteroposterior.
- "The final dataset had 17 006 male patients, 12 888 female patients, **1639 minors, 28 255 adults**". That is about 5.5% minors.
- "There were 12 274 unique patients ... in the 30 000 examination dataset, so several patients had several different examinations, with 75 radiographs being the most for a patient."
- 106 of the 30,000 exams were excluded from annotation (wrong body part, lateral view, or chest cut off).
- No exact age range is given in the paper. Compute it from the DICOM headers.

Consequences for us:

- Do not write "RSNA contains only adults". Write "we kept patients aged 18 or older, using the age in the DICOM header".
- The minors in RSNA are a small free bonus: same hospital and scanner pipeline, different age. Scoring them separately helps tell "age shift" from "hospital shift". They are not aged 1 to 5 like Kermany, so say so.
- Kermany is anterior-posterior only. RSNA is a mix of AP and PA. View is a second shift on top of age and hospital.

### 2.3 Number of labelled images and class counts

**[V]** Counted from `stage_2_detailed_class_info.csv` and `stage_2_train_labels.csv` (the Kaggle label files; read from a public copy on Hugging Face):

| Class | Images | Share |
|---|---|---|
| Lung Opacity (Target = 1) | 6,012 | 22.5% |
| No Lung Opacity / Not Normal (Target = 0) | 11,821 | 44.3% |
| Normal (Target = 0) | 8,851 | 33.2% |
| **Total labelled training images** | **26,684** | |

- The label file has 30,227 rows because an image with several boxes has several rows. There are 9,555 bounding boxes on the 6,012 positive images.
- No image has more than one class.
- The Kaggle stage 2 test set has 3,000 more images. Their labels are not in the Kaggle CSV files.
- Baselines: "always predict no opacity" scores 77.5% on all three classes. If we keep only Lung Opacity versus Normal, "always predict Normal" scores 59.6% (8,851 of 14,863).
- Source files: https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset/tree/main (third-party copy of the Kaggle files; cite RSNA, not this mirror)

**Split warning. [V]** Shih et al.: "Random unique identifiers were generated for each of the 30 000 examinations." So the Kaggle column `patientId` identifies an exam, not a person, and one person can have up to 75 exams. For the learning curve (fine-tuning on N adult images), a random split can leak patients. RSNA publishes a mapping from its IDs back to the NIH files, which contain the real patient number:

- https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json (17.6 MB)

If we do not use the mapping, state the leakage risk as a limitation.

### 2.4 Who labelled, what the three classes mean, and how this relates to pneumonia

**[V]** All from Shih et al. 2019:

- Labellers: "Eighteen board-certified radiologists from 16 academic institutions", mean 10.6 years of experience. Six represented RSNA, twelve the Society of Thoracic Radiology (STR). Tool: MD.ai.
- Original label list: "Opacity (High Probability)", "Opacity (Medium Probability)", "Opacity (Low Probability)", "No Opacity/Not Normal", "Normal".
- Definition given to readers: "Lung opacity (bounding box) was considered to be an area more opaque than the surrounding area ... on a chest radiograph that, in a patient with cough and fever, has a low, medium, or high likelihood of being pneumonia."
- "In the absence of clinical information, lateral radiographs, and serial examinations, readers were required to make assumptions."
- "Readers were advised to exclude obvious masses, nodules, lobar collapse, and linear atelectasis."
- "Cases labeled 'No Opacity/Not Normal' may have lung opacity, but no opacity suspicious for pneumonia."
- Final version used on Kaggle: low-probability boxes removed, medium and high merged into one class.
- Only 4,527 exams were read by three radiologists (two STR, one RSNA, majority vote, 30% individually adjudicated). The others were read by one radiologist. The paper names this as a limitation.
- The paper's own summary: the dataset has "bounding boxes for pulmonary opacity on chest radiographs which may represent pneumonia in the appropriate clinical setting."

Plain-language meaning of the classes:

| Class | Meaning |
|---|---|
| Normal | Radiologist sees nothing abnormal |
| No Lung Opacity / Not Normal | Something is abnormal, but nothing that looks like pneumonia. Other opacities are possible |
| Lung Opacity | At least one opacity that could be pneumonia in a patient with cough and fever |

**The limitation to state in the write-up:** "Lung Opacity" is a radiologist's impression from one image. It is not a confirmed pneumonia diagnosis. There are no symptoms, lab results or follow-up. Kermany labels are also image readings, but by different doctors with a different definition. So part of any performance gap is a change in what the label means, not only a change in patients. The 2026 Applied Sciences paper says the same: the RSNA test "combines age, institutional, acquisition, and label shifts" (section 3.2).

### 2.5 Data-use terms

**[V]** RSNA publishes a one-page "Pneumonia Detection Challenge Terms of Use and Attribution". Full relevant text:

> "You may access and use these de-identified imaging datasets and annotations ("the data") for the purposes of academic research and education, and other commercial or non-commercial purposes as long as you agree to abide by the following provisions:
> 1. Not to make any attempt to identify or contact any individual(s) who may be the subjects of the data.
> 2. If you share or re-distribute the data in any form, to meet the attribution requirements described below"

Attribution for the NIH images: "Provide a link to the NIH download site: https://nihcc.app.box.com/v/ChestXray-NIHCC", "Include a citation to the CVPR 2017 paper" (Wang et al.), and "Acknowledge that the NIH Clinical Center is the data provider".

Attribution for the RSNA annotations: "Provide a link to this download site" (the RSNA challenge page) and "Include a citation to the Radiology: AI 2019 paper" (Shih et al.).

- Terms PDF: https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf?rev=db1621c90e7d4814b070161153cc600b&hash=03B32453B8646245DE131339F09282D1
- Same sentence on the challenge page: https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

Answers to our three questions (our reading of the text, not legal advice):

| Question | Answer | Why |
|---|---|---|
| (a) Use it in a hackathon project? | Yes | "academic research and education, and other commercial or non-commercial purposes" |
| (b) Show a few example images in a public demo? | Yes, with attribution | Sharing "in any form" is allowed if the attribution list is met. Never try to identify a patient |
| (c) Re-upload resized copies? | Yes under the RSNA terms, with attribution | Same clause. Put the two links, the two citations and the NIH acknowledgement next to the upload |

Three caveats:

- **[V]** RSNA's own two-page dataset description lists "Non-commercial purpose" under "Data use agreement/licensing". This contradicts the "commercial or non-commercial" wording of the terms page. Our use is non-commercial, so it does not affect us.
  - https://www.rsna.org/-/media/Files/RSNA/Education/AI%20resources%20and%20training/AI%20image%20challenge/RSNA-2018-Pneumonia-Detection-Challenge-Dataset-Description.ashx?la=en&hash=8331DDFA52AA5EE9C3C99B5983BB680D7BD2ABBF
- **[UNVERIFIED]** If we download from Kaggle, the Kaggle competition rules (section 7, "Competition Data") also apply to that copy. The rules page could not be opened from this session (captcha). A Hugging Face mirror labels the Kaggle zip "subject-to-competition-rules". Either read that section before re-uploading anything, or download from RSNA's own links, which are covered by the terms quoted above.
- RSNA's own download of the challenge images is 3.98 GB of DICOM files: https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip

---

## 3. Prior work on this failure

### 3.1 The papers named in the brief

**Zech et al. 2018, PLOS Medicine. [V]**
Pneumonia CNNs trained at one hospital system did worse at another in 3 of 5 comparisons. A model trained on pooled NIH and Mount Sinai data had AUC 0.931 internally and 0.815 on Indiana University data (P = 0.001). The data were 158,323 chest X-rays from three US hospital systems, all adults (mean ages 46.9, 63.2 and 49.6 years). The networks could also tell which hospital an image came from.

- https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
- Full text used: https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/

**Roberts et al. 2021, Nature Machine Intelligence. [V] for the abstract, [V] for the quote in the arXiv version.**
Systematic review: 2,212 studies found, 415 screened in, 62 reviewed in depth, and "none of the models identified are of potential clinical use due to methodological flaws and/or underlying biases".

They name the Kermany dataset explicitly. Quote (arXiv version 4, January 2021):

> "Many papers (16/61) use the pneumonia dataset of Kermany et al. as a control group. They commonly fail to mention that this consists of paediatric patients aged between one and five. Developing a model using adult COVID-19 patients and very young pneumonia patients is likely to overperform as it is merely detecting children vs. adults. This dataset is also erroneously referred to as the Mooney dataset in many papers (being the Kermany dataset deployed on Kaggle)."

And in the recommendations: "including paediatric images in the COVID-19 context introduces a strong bias."

- Abstract and numbers: https://api.crossref.org/works/10.1038/s42256-021-00307-0
- Quote: https://arxiv.org/abs/2008.06388
- Journal page: https://www.nature.com/articles/s42256-021-00307-0 (returned "too many requests" in this session; the count reads 16/61 in the arXiv text and the journal version reviews 62 papers, so check the fraction on the journal page before quoting it)

**DeGrave, Janizek and Lee 2021, Nature Machine Intelligence. [V] for the abstract of the preprint.**
"recent deep learning systems to detect COVID-19 from chest radiographs rely on confounding factors rather than medical pathology, creating an alarming situation in which the systems appear accurate, but fail when tested in new hospitals." The exact AUC drop was not re-checked (section 7).

- https://www.nature.com/articles/s42256-021-00338-7
- Preprint abstract: https://doi.org/10.1101/2020.09.13.20193565

**Geirhos et al. 2020, Nature Machine Intelligence. [V]**
Perspective article that gives the name: "Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios." It has no single key number.

- https://www.nature.com/articles/s42256-020-00257-z
- https://arxiv.org/abs/2004.07780

**Maguolo and Nanni 2021, Information Fusion. [V]**
Classifiers trained on X-rays with the centre blacked out still separate the classes, so "the neural networks are learning patterns in the dataset that are not correlated to the presence of COVID-19". Roberts et al. cite this for the Kermany and Cohen sources.

- https://doi.org/10.1016/j.inffus.2021.04.008

### 3.2 Papers that tested a Kermany-trained model on adults

**Javier-Tarapaqui et al., Applied Sciences, published 9 November 2026. [V] abstract.**
Nine architectures and an ensemble trained on the Kermany children's data, best internal accuracy 98.18%. Then: "An external evaluation on an adult RSNA subset, which combines age, institutional, acquisition, and label shifts, revealed markedly different behavior: external accuracy ranged from 71.30% to 86.20%, and VGG16, among the most accurate models internally, was the least robust." They also report calibration and quantitative Grad-CAM.

- https://doi.org/10.3390/app162111063
- **[UNVERIFIED]** A search snippet says VGG16 called 801 of 1,000 normal adult X-rays pneumonia (specificity 19.9%), and that the authors note age is perfectly confounded with dataset. The full text could not be opened here.

**Gómez-Vela et al., Applied Intelligence, September 2026. [V] abstract.**
"A distinctive methodological feature of this study is the use of two demographically distinct datasets: a pediatric chest X-ray dataset for training, and an independent adult population dataset for external validation." Focus is on which architectures keep performance and explanation quality. The abstract gives no numbers and does not name the adult dataset.

- https://doi.org/10.1007/s10489-026-07398-5

**Neo, BMC Research Notes, May 2026. [V] full text.**
ResNet-18 trained on PneumoniaMNIST (Kermany images shrunk to 28 x 28): AUC 0.9735 on its own test set, **0.7458** with no adaptation on a balanced 1,000-image RSNA subset. After "brief fine-tuning" on 400 RSNA images with a frozen backbone: **0.763**. So a small amount of adult data closed almost none of the gap in that setup.

- https://doi.org/10.1186/s13104-026-07793-8
- Full text: https://europepmc.org/article/PMC/PMC13330209

Unverified leads from search results, not opened: an SSRN preprint by K. Promjiraprawat on "external threshold-transfer" for Kermany-trained CNNs (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7305448) and a GitHub project `t-Shack/Pneumonia_AI`.

### 3.3 Candid answer: how original is our question?

- "Does a Kermany model fail on RSNA adults?" has been answered in print, three times in 2026, with the same pair of datasets. The answer is yes: accuracy falls from about 98% to between 71% and 86%, or AUC from 0.97 to 0.75.
- The cause (children versus adults) was named in 2021 by Roberts et al.
- Do not present the drop as a discovery. Present it as a replication, cite these papers, and say what we add.

What we found no paper doing (absence in a short search, not proof):

- A learning curve: accuracy on adults as a function of the number of adult images added. Neo 2026 has one point (400 images, frozen backbone).
- A warning for out-of-distribution inputs built into a usable demo.
- A split of the errors inside RSNA by age (including the minors), sex and AP/PA view, to see which shift hurts most.
- The size shortcut in section 1.5 as an explicit baseline.

---

## 4. Impact numbers

### 4.1 Pneumonia deaths

| Statistic | Year of data | Status | Source |
|---|---|---|---|
| 740,180 children under 5 died of pneumonia; 14% of all deaths under 5; 22% of deaths at ages 1 to 5 | 2019 | [V] | WHO fact sheet "Pneumonia in children", 11 November 2022 |
| "more than 725,000 children under the age of 5" die every year, "including around 190,000 newborns"; "Every 43 seconds at least one child dies from pneumonia" | year not stated on the page | [V] | UNICEF explainer, 11 November 2025 |
| 2.18 million deaths from lower respiratory infections, all ages (95% interval 1.98 to 2.36 million) | 2021 | [V] | GBD 2021, Lancet Infectious Diseases 2024 |
| 502,000 of these in children under 5 (406,000 to 611,000) | 2021 | [V] | same |
| 1.11 million of these in adults aged 70 and older (978,000 to 1.20 million); highest death rate of any age group, 224.6 per 100,000 | 2021 | [V] | same, Table 2 |
| "only one third of children with pneumonia receive the antibiotics they need" | not stated | [V] | WHO fact sheet |

- WHO: https://www.who.int/news-room/fact-sheets/detail/pneumonia
- UNICEF: https://www.unicef.org/stories/childhood-pneumonia-explained
- GBD 2021: https://doi.org/10.1016/S1473-3099(24)00176-2 and full text https://pmc.ncbi.nlm.nih.gov/articles/PMC11339187/

Do not mix the numbers. WHO counts pneumonia in 2019. GBD counts "lower respiratory infections", defined as pneumonia or bronchiolitis and excluding COVID-19, in 2021. That is why the under-5 figures differ (740,180 versus 502,000).

Useful for our story: in 2021 more than twice as many people aged 70 and older died of these infections as children under 5, yet the standard tutorial dataset contains only children aged 1 to 5.

### 4.2 Radiologist shortage

**[V]** Kawooya et al. 2022, *Insights into Imaging*, Table 3 ("Numbers of imaging personnel for selected African countries"):

| Country | Radiologists | Population | Year | Per million people (our division) |
|---|---|---|---|---|
| Malawi | 3 | 19,681,633 | 2021 | 0.15 |
| Mozambique | 5 | 29,500,000 | 2018 | 0.17 |
| D.R. Congo | 42 | 84,070,000 | 2018 | 0.50 |
| Uganda | 55 | 42,460,000 | 2021 | 1.30 |

- The same paper: "over 90% of radiologists in Africa are working in urban or periurban settings".
- Warning: for Kenya, Tanzania, Nigeria and Ethiopia the ratio column in that table does not match radiologists divided by population. Only the four rows above are internally consistent. Do not quote the others.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8959275/ (doi:10.1186/s13244-022-01203-w)

**[V]** Rich-country comparison, Royal College of Radiologists 2024 census: "The UK has a 29% shortfall of consultant clinical radiologists. This is projected to rise to 39%, should no action be taken." As of September 2024 the UK "needs 1,953 more clinical radiology consultants". "100% of Clinical Directors reported concerns about backlogs and delays becoming worse."

- https://www.rcr.ac.uk/media/4imb5jge/_rcr-2024-clinical-radiology-workforce-census-report.pdf

### 4.3 Does a delayed reading matter?

- **[V]** Annarumma et al. 2019, *Radiology*: an AI triage system for adult chest X-rays, tested in a simulation on 15,887 images at a London hospital trust. "The average reporting delay was reduced from 11.2 to 2.7 days for critical imaging findings". This shows that reports for critical chest X-rays waited 11.2 days on average in the historical data. It is a simulation, not a trial.
  - https://doi.org/10.1148/radiol.2018180921
- **[V]** Houck et al. 2004, *Archives of Internal Medicine*: 13,771 US patients over 65 with pneumonia. Antibiotics within 4 hours of arrival went with lower 30-day mortality (11.6% versus 12.7%, adjusted odds ratio 0.85, 95% interval 0.76 to 0.95).
  - https://doi.org/10.1001/archinte.164.6.637
  - Be honest about this one: it is an observational study about antibiotic timing, not about X-ray reading. It supports "speed matters in pneumonia", nothing more specific.

No source was found that directly measures harm from slow chest X-ray reading in pneumonia.

---

## 5. A third, independent test dataset

### 5.1 `HANTIFARAH/chest-xray-pneumonia-epic-chittagong`: do not use as independent data

**What the record claims. [V]**

- The Hugging Face repository has no description and no licence. Splits: train 1,823, validation 394, test 393 (2,610 images, 1,304 normal and 1,306 pneumonia). Created 28 May 2026.
  - https://huggingface.co/datasets/HANTIFARAH/chest-xray-pneumonia-epic-chittagong
- Its apparent origin is a Mendeley Data record, "A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh", by Hira MIK, Bithee MMA and others (National Institute of Textile Engineering and Research; University of Dhaka). CC BY 4.0. Six versions, October 2025 to October 2026. doi:10.17632/wndbd5r26y.
  - https://data.mendeley.com/datasets/wndbd5r26y/6
- Versions 1 to 5 state 3,355 "primary" images "raw and unprocessed". No patient ages, no hospital department, no equipment, no ethics statement.
- Version 6 (25 October 2026) is a cleaned release, and its own text raises the alarm:
  - "The Version 5 page text stated 3,355 images, but the Version 5 archive contained 2,626."
  - "96 images belonging to groups of identical images with conflicting class labels were excluded."
  - "Labels come from the source folder structure and are not independently clinically adjudicated".
  - "No patient identifiers exist, so splits are image-level and patient independence is unknown."
  - "Ethics approval, consent and acquisition details: [AUTHOR TO FILL]."

**What we measured. [OWN]** Image sizes for all 2,610 Hugging Face images, compared with all 5,856 Kermany images:

| Group in the "Chittagong" set | Images | Finding |
|---|---|---|
| "normal", square 1024 x 1024 | 513 | Adult films. 1024 x 1024 is the NIH ChestX-ray14 format |
| "normal", square 512 x 512 | 507 | Adult films |
| "normal", square 256 x 256 | 78 | Adult films |
| "normal", other sizes | 206 | Every size also occurs among Kermany NORMAL images |
| "pneumonia", all sizes | 1,306 | Every size also occurs in Kermany; only one image is square |

- All 1,387 distinct non-square sizes in this set also occur in Kermany. Kermany has 4,803 distinct sizes, so this cannot be chance.
- We downloaded 8 pairs with a size that occurs exactly once in each dataset. **All 8 are the same picture**: pixel correlation between 0.9998 and 1.0000, mean difference under 0.7 grey levels out of 255. Looking at them confirms small children.
- In 2 of the 8 pairs, a Kermany NORMAL image carries the label "pneumonia" in the "Chittagong" set. By size alone, about 330 of the 1,306 "pneumonia" images match only Kermany NORMAL images. That is an estimate; two cases are confirmed by pixels.
- The square "normal" images are visibly adults. Some carry "PORTABLE" and "AP" markers.

Conclusion: in the Hugging Face copy, the pneumonia class is Kermany children (partly mislabelled) and most of the normal class is adult films from elsewhere. A model can score well by telling children from adults or square from non-square images. It is neither independent nor a valid test.

Limits of this check: we tested the Hugging Face copy, not the 524 MB Mendeley archive. The image counts are close (2,610 versus 2,626) and version 6 reports the same kind of label conflicts, so the archive is very probably the same material, but that is not proven. The origin of the adult images was not traced.

How to reproduce: page through `https://datasets-server.huggingface.co/rows?dataset=<id>&config=default&split=<split>&offset=<n>&length=100` for both datasets, collect `width` and `height`, match sizes that occur once in each, download those image pairs from the `src` links, and compare pixels.

### 5.2 Other candidates checked

| Candidate | Origin and ages | Images | Licence and access | Independent of Kermany, RSNA, NIH? |
|---|---|---|---|---|
| `hf-vision/chest-xray-pneumonia` (HF) | Kermany | 5,216 / 16 / 624 | CC BY 4.0 | No. It is Kermany. [V] |
| `mmenendezg/pneumonia_x_ray` (HF) | Card: "processed version of the original" UCSD dataset | 4,187 / 1,045 / 624 | none stated | No. Kermany. [V] |
| `HANTIFARAH/chest-xray-pneumonia` (HF) | Same split sizes as the "Chittagong" set | 1,823 / 394 / 393 | none | No, presumably the same files. Not checked by pixel |
| `realsudarshan/chest-xray-tb-pneumonia` (HF) | Card: "Derived from public sources (NIH Chest X-ray Dataset, RSNA Pneumonia Detection Challenge, Kaggle TB datasets)" | not checked | CC BY 4.0 | No. [V] |
| `opencampus/chest-xray-pneumonia-3class-balanced` (HF) | Classes NORMAL, BACTERIAL, VIRAL, which are the Kermany file-name classes | not checked | CC BY 4.0 | Almost certainly Kermany. Not confirmed |
| Mendeley `8cm4ffk7b4`, "Pneumonia Chest X-ray Dataset (2023-2024)" | "a local hospital in Pakistan", unnamed. Ages not given | 1,345, pneumonia only | CC BY 4.0, free | Unknown. No normals, no ages, no linked paper. Run the size check before any use |
| Mendeley `m4s2jn3csb`, "Chest X-Ray Image" | "various hospitals in Bangladesh". Ages not given | 4,350 in four classes (Normal 1,200; Lung Opacity 1,100; COVID; viral pneumonia) | CC BY 4.0, free | Unknown, doubtful. The class names match a well-known mixed COVID database built from Kermany and RSNA. Run the size check first |
| Mendeley `3tx8xymdsv`, "Child Pneumonia Dataset" | Sirajganj, Bangladesh. Children. Phone photos of X-ray films | 282 | CC BY 4.0, free | Probably. Too small, photos not scans, no normal class described |
| Cohen et al., covid-chestxray-dataset (GitHub) | Collected from publications and teaching sites; Germany, Italy, UK, Australia, Spain and others. **Ages 18 to 94** where recorded, no minors | 782 frontal X-rays from 449 patients: 478 COVID-19, 191 other pneumonia, only 18 "No Finding" | Free, no request. Licence differs per image (mostly CC BY-NC-SA and CC BY 3.0; 223 rows blank) | **Yes.** [V] But almost no normal images |
| VinDr-PCXR (PhysioNet) | Vietnam. Children under 10 | 9,125 (7,728 train, 1,397 test) | Free account plus signing a data-use agreement. No sharing of images | Yes. [V] Children, not adults |
| VinDr-CXR (PhysioNet) | Hospital 108 and Hanoi Medical University Hospital, Vietnam. Adults | 18,000 | "Only credentialed users who sign the DUA can access the files" | Yes, but needs manual credentialing. [V] |
| CheXpert, MIMIC-CXR, PadChest | Stanford; Boston; Alicante, Spain. Adults | large | Registration or signed agreement required | Yes, but access conditions were not re-checked here |

- Mendeley records: https://data.mendeley.com/datasets/8cm4ffk7b4/1 , https://data.mendeley.com/datasets/m4s2jn3csb/1 , https://data.mendeley.com/datasets/3tx8xymdsv/1
- Cohen: https://github.com/ieee8023/covid-chestxray-dataset (counts are ours, from `metadata.csv`)
- VinDr-PCXR: https://physionet.org/content/vindr-pcxr/1.0.0/
- VinDr-CXR: https://physionet.org/content/vindr-cxr/1.0.0/

Unofficial copies of VinDr, PadChest and CheXpert exist on Hugging Face without a gate (for example `sunday-hao/vindr-cxr-testset`, `JasonZZ0601/PadChest-GR-Xray`, `danjacobellis/chexpert`). The original owners require a signed agreement, so these copies are probably not allowed. Using them would be a citation and licence problem with the judges. Avoid.

### 5.3 Bottom line

No dataset was found that meets all three conditions and has both normal and pneumonia adults. Honest options, in order of effort:

1. **No third dataset.** Use RSNA as the one external test. Add the split inside RSNA by age group, sex and view. This costs nothing extra and is clean.
2. **Cohen dataset as an extra check on adults with pneumonia only.** It gives sensitivity ("how many of these 669 pneumonia X-rays from many countries does the model flag?"), not specificity. Its README warns: "please do not claim diagnostic performance of a model without a clinical study". Mixed licences: do not re-upload the images.
3. **VinDr-PCXR as a second children's hospital.** Kermany to VinDr-PCXR changes the hospital but not the age group; Kermany to RSNA changes both. Needs a PhysioNet account, a signed agreement, DICOM handling, and the images may not be shown publicly. Probably too much for one weekend.

Before using any "new" pneumonia dataset from Hugging Face, Kaggle or Mendeley, run the size check from section 5.1.

---

## 6. How widely used is the Kermany dataset?

| Number | Value | Status | Source |
|---|---|---|---|
| Kaggle downloads | 715,431 | [V] | Kaggle public API, 14 November 2026 |
| Kaggle views | 3,261,789 | [V] | same |
| Kaggle upvotes | 7,775 | [V] | same |
| Kaggle notebooks using it (`kernelCount`) | 4,066 | [V] | same |
| Citations of the Cell paper, Crossref | 4,400 | [V] | Crossref |
| Citations of the Cell paper, OpenAlex | 5,093 (646 so far in 2026) | [V] | OpenAlex |

- https://www.kaggle.com/api/v1/datasets/list?search=chest-xray-pneumonia&user=paultimothymooney
- https://api.crossref.org/works/10.1016/j.cell.2018.02.010
- https://api.openalex.org/works/https://doi.org/10.1016/j.cell.2018.02.010

Caveat on citations: the Cell paper is mainly about eye scans (OCT). Not every citation concerns the chest X-rays. The Kaggle numbers are the better evidence for "standard tutorial dataset".

Two more signs that it is the default:

- **[V]** It is built into the MedMNIST benchmark: "The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images." (Yang et al. 2023, *Scientific Data*)
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/
- **[V]** The official Keras example "Pneumonia Classification on TPU" (created 28 July 2020) loads `gs://download.tensorflow.org/data/ChestXRay2017/` and predicts NORMAL versus PNEUMONIA. That "ChestXRay2017" is the Kermany release is our inference from the name and classes; the page does not cite the paper.
  - https://keras.io/examples/vision/xray_classification_with_tpus/
- **[V]** Roberts et al. found it used as the control group in 16 of the COVID-19 papers they reviewed in depth (section 3.1).

---

## 7. Not verified / uncertain

Sources that could not be opened from this session: cell.com, nature.com (rate limit), mdpi.com, link.springer.com, papers.ssrn.com, data.unicef.org, the Kaggle web pages (the public Kaggle API did work), and github.com outside the Cohen raw files.

1. **Numbers in the Kermany paper itself.** From memory, the paper reports 5,232 training images (3,883 pneumonia, of which 2,538 bacterial and 1,345 viral; 1,349 normal) and 624 test images (234 normal, 390 pneumonia) "from 624 patients", and about 92.8% accuracy for pneumonia versus normal. The totals agree with the file counts in section 1.3, but the wording and the accuracy were not re-read.
2. **Roberts et al. fraction.** "16/61" is the arXiv version 4 text. The journal version reviews 62 papers; check whether it prints 16/62.
3. **DeGrave et al. numbers.** Only the preprint abstract was read. No AUC values were verified.
4. **Applied Sciences 2026 details** beyond the abstract (801 of 1,000 normals misclassified by VGG16; the age-confounding remark; how the RSNA subset was drawn).
5. **Which adult dataset** Gómez-Vela et al. 2026 used, and their numbers.
6. **Kaggle competition rules, section 7**, for RSNA data downloaded through Kaggle.
7. **NIH's own wording** of its usage terms and the NIH age range. The attribution requirements in section 2.5 are RSNA's restatement. Whether the NIH age field contains implausible values should be checked in our own data.
8. **Whether the patient-ID overlap in Kermany is real** (same children in train and test) or only a numbering artefact.
9. **Whether a trained network uses the size shortcut** in section 1.5. We showed the shortcut exists, not that it is used.
10. **The Mendeley "Chittagong" archive itself.** Only the Hugging Face copy was fingerprinted. The source of its adult "normal" images was not traced. The figure of about 330 mislabelled images is a size-based estimate.
11. **The Pakistani and Bangladeshi Mendeley datasets** in section 5.2. Our doubts rest on class names and counts, not on a pixel check.
12. **Pneumonia label counts in VinDr-PCXR**, and whether its data-use agreement is granted instantly.
13. **Access conditions for CheXpert, MIMIC-CXR and PadChest** were not re-checked.
14. **The often-quoted claim that two thirds of the world has no access to basic radiology** (attributed to WHO/PAHO) and the **radiologists-per-million figures of the Lancet Oncology Commission 2021**. No primary source was opened. Do not use them.
15. **AP versus PA and the "Lung Opacity" rate in RSNA.** AP films are usually taken of sicker, bedridden patients, so the opacity rate is expected to differ by view. Not checked here; compute it from the DICOM headers.
16. **"We found no paper doing X"** in section 3.3 is the result of a short search. Phrase it as "to our knowledge".
17. The search results `t-Shack/Pneumonia_AI` (GitHub) and the SSRN preprint were not opened.

---

## 8. Citations (ready to paste)

### Datasets

- Kermany D, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, V2, 2018. doi:10.17632/rscbjbr9sj.2. Licence CC BY 4.0. https://data.mendeley.com/datasets/rscbjbr9sj/2
- Mooney P. Chest X-Ray Images (Pneumonia). Kaggle, 2018 (copy of the dataset above). https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Radiological Society of North America. RSNA Pneumonia Detection Challenge (2018). Images and annotations. https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
- NIH Clinical Center. NIH Chest X-ray Dataset (ChestX-ray8 / ChestX-ray14). https://nihcc.app.box.com/v/ChestXray-NIHCC . The NIH Clinical Center is the data provider.
- Cohen JP, Morrison P, Dao L, Roth K, Duong TQ, Ghassemi M. COVID-19 Image Data Collection: Prospective Predictions Are the Future. arXiv:2006.11988, 2020. https://github.com/ieee8023/covid-chestxray-dataset (only if used)
- Hira MIK, Bithee MMA, et al. A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh. Mendeley Data, 2025-2026. doi:10.17632/wndbd5r26y.6. Hugging Face copy: HANTIFARAH/chest-xray-pneumonia-epic-chittagong. (examined and rejected, see section 5.1)

### Papers on the datasets

- Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018;172(5):1122-1131.e9. doi:10.1016/j.cell.2018.02.010
- Shih G, Wu CC, Halabi SS, et al. Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia. Radiology: Artificial Intelligence. 2019;1(1):e180041. doi:10.1148/ryai.2019180041
- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR. 2017:3462-3471. doi:10.1109/CVPR.2017.369
- Yang J, Shi R, Wei D, et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data. 2023;10:41. doi:10.1038/s41597-022-01721-8

### Papers on generalisation and shortcuts

- Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine. 2018;15(11):e1002683. doi:10.1371/journal.pmed.1002683
- Roberts M, Driggs D, Thorpe M, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nature Machine Intelligence. 2021;3(3):199-217. doi:10.1038/s42256-021-00307-0
- DeGrave AJ, Janizek JD, Lee S-I. AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence. 2021;3(7):610-619. doi:10.1038/s42256-021-00338-7
- Geirhos R, Jacobsen J-H, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. Shortcut learning in deep neural networks. Nature Machine Intelligence. 2020;2(11):665-673. doi:10.1038/s42256-020-00257-z
- Maguolo G, Nanni L. A critic evaluation of methods for COVID-19 automatic detection from X-ray images. Information Fusion. 2021;76:1-7. doi:10.1016/j.inffus.2021.04.008
- Javier-Tarapaqui J, Vega-Huerta H, Guerra-Grados L, et al. Deep Learning Architecture Comparison for Pneumonia Detection in Pediatric Chest X-Rays: Ensemble Learning and Cross-Population External Evaluation. Applied Sciences. 2026;16(21):11063. doi:10.3390/app162111063
- Gómez-Vela F, López-Fernandez A, Divina F, García-Torres M. Performance-interpretability trade-offs and generalization in deep learning for pneumonia detection: A benchmarking study. Applied Intelligence. 2026;56(14):414. doi:10.1007/s10489-026-07398-5
- Neo RYS. Benchmarking knowledge distillation for lightweight pneumonia detection: a multi-seed calibration study on PneumoniaMNIST. BMC Research Notes. 2026;19(1):270. doi:10.1186/s13104-026-07793-8
- Shahi K, Bagale A. Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations. arXiv:2511.00456, 2025.

### Impact sources

- World Health Organization. Pneumonia in children (fact sheet). 11 November 2022. https://www.who.int/news-room/fact-sheets/detail/pneumonia
- UNICEF. Pneumonia in children: What you need to know. 11 November 2025. https://www.unicef.org/stories/childhood-pneumonia-explained
- GBD 2021 Lower Respiratory Infections and Antimicrobial Resistance Collaborators. Global, regional, and national incidence and mortality burden of non-COVID-19 lower respiratory infections and aetiologies, 1990-2021: a systematic analysis from the Global Burden of Disease Study 2021. Lancet Infectious Diseases. 2024;24(9):974-1002. doi:10.1016/S1473-3099(24)00176-2
- Kawooya MG, Kisembo HN, Remedios D, et al. An Africa point of view on quality and safety in imaging. Insights into Imaging. 2022;13:58. doi:10.1186/s13244-022-01203-w
- The Royal College of Radiologists. Clinical Radiology Workforce Census 2024. https://www.rcr.ac.uk/media/4imb5jge/_rcr-2024-clinical-radiology-workforce-census-report.pdf
- Annarumma M, Withey SJ, Bakewell RJ, Pesce E, Goh V, Montana G. Automated Triaging of Adult Chest Radiographs with Deep Artificial Neural Networks. Radiology. 2019;291(1):196-202. doi:10.1148/radiol.2018180921
- Houck PM, Bratzler DW, Nsa W, Ma A, Bartlett JG. Timing of antibiotic administration and outcomes for Medicare patients hospitalized with community-acquired pneumonia. Archives of Internal Medicine. 2004;164(6):637-644. doi:10.1001/archinte.164.6.637
