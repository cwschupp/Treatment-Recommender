Preliminary Proposal (Clayton Schupp)
===

Let's Help Scratch That Itch
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
Psoriasis.  The data is collected directly from the treating doctor and is 
updated instantaneously in the cloud.  This allows a unique opportunity to
use crowd-sourced data to guide dermatologists in their treatment choice.

The feature variables of interest are those that would be collected on 
an initial examination of a new patient.  In addition to demographic and 
behavioral data collected at intake such as age, gender, race, smoking status;
a new patient would present with characteristics specific to psoriasis.  These features would include some if not all of the following: symptom 
severity and percent of surface body area affected, overall severity.  These
variables are represent the common confounders and risk factors for Psoriasis
in the literature and normally adjusted for in statistical analyses.

In this project, I select patients that have been diagnosed with Psoriasis
within the past 3 years, cluster them into homogenous groups via the feature
variables. Then within each cluster, I identify the treatments used and for 
each treatment, model the relationship with change in severity and reduction
in body surface affected.

My main model is cluster_model in which I take tab delimited data, read 
it into pandas and return dataframes that contain: continuous variables,
categorical variables, outcome variables, and treatment.  I then do some
preprocessing of the features to scale the continuous variables and create
indicator variables for the categorcal variables.  I then run KMeans, 
implementing the gap statistic to identify the optimal number of clusters.  My
program returns a dictionary of all the cluster level variables including
Plotly calls to separate barchart and linechart programs to produce the 
treatment and outcome plots. These plots are stored in the cloud and their
unique url is maintained in the cluster level dictionary of results.

Then in my web app, I have some basic html to have a user enter a new patient's
information which calls on my model to classify the individual to a cluster.
The treatment and outcome plots are then returned for that patient and would 
identify the 3 most commonly used treatments and that treatment's 
effectiveness.



