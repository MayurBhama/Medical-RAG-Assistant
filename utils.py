from Bio import Entrez

Entrez.email = "mayurbhama999@gmail.com"

def fetch_pubmed_abstracts(query , max_results=20):
    #Getting IDs of relevant Papers 
    handle = Entrez.esearch(db = "pubmed", term = query, retmax = max_results, sort = "relevance")
    record = Entrez.read(handle)
    handle.close()

    id_list = record["IdList"]

    if not id_list:
        return "No papers found for this topic"

    ids = ",".join(id_list)
    handle = Entrez.efetch(db= "pubmed", id = ids, rettype= "abstract", retmode = "text")
    papers_text = handle.read()
    handle.close()

    return papers_text