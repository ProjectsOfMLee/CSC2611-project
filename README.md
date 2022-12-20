# CSC2611-project
This is the Repo for Computational Semantic Change Course Capstone Project.

Full Presentation Slides: https://docs.google.com/presentation/d/1J9_k48yt2Msn836-zUJ1p2kl1bj13QsNrtHj_hC_9E4/edit?usp=sharing

For the full datasets and trained word embeddings / models, contact: mhorusli@cs.toronto.edu
# Previous References (Deprecated)

1. Learning Dynamic Contextualised Embedding (2022) https://arxiv.org/pdf/2208.10734.pdf 
2. Di Carlo, V., Bianchi, F., & Palmonari, M. (2019). Training Temporal Word Embeddings with a Compass. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 6326-6334. https://doi.org/10.1609/aaai.v33i01.33016326 
3. @inproceedings{jawahar-seddah-2019-contextualized,
    title = "Contextualized Diachronic Word Representations",
    author = "Jawahar, Ganesh  and
      Seddah, Djam{\'e}",
    booktitle = "Proceedings of the 1st International Workshop on Computational Approaches to Historical Language Change",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-4705",
    doi = "10.18653/v1/W19-4705",
    pages = "35--47",
    abstract = "Diachronic word embeddings play a key role in capturing interesting patterns about how language evolves over time. Most of the existing work focuses on studying corpora spanning across several decades, which is understandably still not a possibility when working on social media-based user-generated content. In this work, we address the problem of studying semantic changes in a large Twitter corpus collected over five years, a much shorter period than what is usually the norm in diachronic studies. We devise a novel attentional model, based on Bernoulli word embeddings, that are conditioned on contextual extra-linguistic (social) features such as network, spatial and socio-economic variables, which are associated with Twitter users, as well as topic-based features. We posit that these social features provide an inductive bias that helps our model to overcome the narrow time-span regime problem. Our extensive experiments reveal that our proposed model is able to capture subtle semantic shifts without being biased towards frequency cues and also works well when certain contextual features are absent. Our model fits the data better than current state-of-the-art dynamic word embedding models and therefore is a promising tool to study diachronic semantic changes over small time periods.",
}
4. Maja Rudolph and David Blei, 2017. Dynamic Bernoulli Embeddings for Language Evolution. arxiv preprint arxiv:1703.08052.
## [evaluation related]
### word embedding direct evaluation method
1. Wang, B., Wang, A., Chen, F., Wang, Y., & Kuo, C. (2019). Evaluating word embedding models: Methods and experimental results. APSIPA Transactions on Signal and Information Processing, 8, E19. doi:10.1017/ATSIP.2019.12 https://www.cambridge.org/core/services/aop-cambridge-core/content/view/EDF43F837150B94E71DBB36B28B85E79/S204877031900012Xa.pdf/div-class-title-evaluating-word-embedding-models-methods-and-experimental-results-div.pdf [grand general guidelines]
2. Arianna Betti, Martin Reynaert, Thijs Ossenkoppele, Yvette Oortwijn, Andrew Salway, and Jelke Bloem. 2020. Expert Concept-Modeling Ground Truth Construction for Word Embeddings Evaluation in Concept-Focused Domains. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6690–6702, Barcelona, Spain (Online). International Committee on Computational Linguistics. https://aclanthology.org/2020.coling-main.586.pdf [philosphy, no code available]
3. https://arxiv.org/pdf/2005.03812.pdf [COMPARATIVE ANALYSIS OF WORD EMBEDDINGS FOR CAPTURING WORD SIMILARITIES]
4. https://arxiv.org/pdf/1605.09096.pdf [histwords]
### use domain specific embedding not as an "end" but applied (e.g. social science prior knowledge/results)
1. https://bigdata.duke.edu/wp-content/uploads/2022/07/Team-19_Data-Plus-Poster_Final_pdf.pdf
2. https://arxiv.org/pdf/1711.08412.pdf
3. https://aclanthology.org/D15-1036.pdf
### use eval methods from lab, visualize & compare the sims/clusters
1. first - last
2. max/min/mean/median/...
3. cluster & jaccard
