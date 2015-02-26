Preliminary Proposal (Clayton Schupp)
===

Scratch That Itch
---

Psoriasis is a common skin condition that changes the life cycle of skin cells
and caused them to build up rapidly on the surface of the skin.  These extra
cells form thick, silvery scales and itchy, dry, red patches that are often 
painful.  It is a persistent, chronic disease that often runs through cycles
of better and worsening symptoms. Althought there isn't a cure, the goal of 
treatment is to stop the cells from growing so quickly and offer significant
relief to the patient.

Through a consulting project on which I am the statistician, I have access 
to electronic medical record data for approximately 25% of all dermatology
patients, corresponding to approximately 250,000 patients suffering with 
psoriasis.  The data is collected directly from the treating doctor and is 
updated instantaneously in the cloud.  This allows a unique opportunity to
use crowd-sourced data to guide treatment

Although I have not yet seen the data, I am currently working with their 
data engineering team to obtain remote access to their big data infrastructure.
Some high level statistics regarding the data available across all disease 
diagnoses collected by the company: 23 Million(M) patient encounters, 13M 
prescriptions, 7M lab results, 180M addresses. I am unsure at this moment 
which features will be of primary interest; however, the goal of this
project is if presented with a new patient, to classify them, and recommend 
the best treatment personalized for them.

In addition to demographic and behavioral data collected at intake such as age, 
gender, race, smoking status, alcohol usage; a new patient would present with 
characteristics specific to psoriasis.  These features would include some if not
all of the following: area of the body affected, type of symptom, symptom 
severity, percent of surface body area affected, overall severity.

Analysis Outline
---

1. Identify an appropriate timeframe to collect the data, I believe data is
available starting 5 or more years ago. The data would be sparse in the 
beginning before more doctors began adopting the technology.  I would 
probably look at the most recent 50-60% and choose a collection start 
date based on that.

2. Collect all intake (first visits) for the patients, identifying what
variable are available for a 'new' patient.

3. Classify these patients using unsupervised learning into clusters.

4. For each patient, collect all subsequent visits and extract treatments
used and change in outcome.  An outcome variable will need to be either
identified or created to measure effectiveness of treatment.

5. For each patient identify the treatment (or combination of treatments) 
that led to the best change in outcome.

6. Within each cluster identify the most effective treatments used.

7. Within each treatment, model change in outcome based on preliminary intake
variables.

8. Return list of treatments in decreasing order of effectiveness

