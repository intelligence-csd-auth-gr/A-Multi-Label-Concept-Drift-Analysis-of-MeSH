# Beyond MeSH Annual Revisions: A Multi-Label Concept Drift Analysis
This repository contains the code for our paper.

## Abstract:
MeSH (Medical Subject Headings) is a hierarchically structured thesaurus used for indexing biomedical information. This vocabulary contains most of the biomedical knowledge available to date. To keep up with the continuous evolution and expanding of our understanding on the medical field, yearly revisions take place in MeSH. These revisions introduce new descriptors in the thesaurus, in addition to changes in already existing ones, either directly or indirectly. This constant evolution of the thesaurus causes many older descriptors to exhibit some form of drift in their meaning, which in turn affects the performance of Machine Learning models trained on an older version of the thesaurus when used to predict data obtained from more recent versions. In this paper, we study the phenomenon of concept drift in MeSH, through evaluating the performance of a state-of-the-art text classification algorithm in articles from different years. We also investigate how changes in descriptors indirectly affect different ones that are related to them by studying the shifts in their co-occurrence, using this shift as a measure of concept drift.


## Files
The data sets used can be found in the following [link](https://drive.google.com/drive/folders/10SvWVJAi7yo1-kZf24BrJzkDaXkfAfD4?usp=sharing), while the predictions from BERT for each year that were used for the Performance based method can be seen [here](https://drive.google.com/drive/folders/11y1tyrA9cNjW2ZGPA30dCs2pezE3gHXU?usp=sharing)

**preliminaries** contains the scripts used to create the final data sets which can be found in the link above from the BioASQ challenge ones.

**Performance based** contains the scripts that perform the performance based analysis using the predictions from the BERT model shared above.

**Co-occurence based** contains the script for the co-occurence based analysis.


## Developed by: 

|           Name  (English/Greek)                    |      e-mail          |
| ---------------------------------------------------| ---------------------|
| Nikolaos Mylonas    (Νικόλαος Μυλωνάς)             | myloniko@csd.auth.gr |
| Ioannis Mollas      (Ιωάννης Μολλάς)               | iamollas@csd.auth.gr |
| Grigorios Tsoumakas (Γρηγόριος Τσουμάκας)          | greg@csd.auth.gr     |

## Funded by

The research work was supported by the Hellenic Foundation forResearch and Innovation (H.F.R.I.) under the “First Call for H.F.R.I.Research Projects to support Faculty members and Researchers and the procurement of high-cost research equipment grant” (ProjectNumber: 514).

## Additional resources

- [AMULET project](https://www.linkedin.com/showcase/amulet-project/about/)
- [Academic Team's page](https://intelligence.csd.auth.gr/#)
 
 ![amulet-logo](https://user-images.githubusercontent.com/6009931/87019683-9204ad00-c1db-11ea-9394-855d1d3b41b3.png)




