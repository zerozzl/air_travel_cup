from elasticsearch import Elasticsearch


def retrieve(es_host, es_port, es_index, es_doc_type, page_size, match_name, match_value):
    es = Elasticsearch([{'host': es_host, 'port': es_port}])

    result = es.search(index=es_index, doc_type=es_doc_type, size=page_size, body={
        'query': {
            'match': {
                match_name: match_value
            }
        }
    })

    hits = result['hits']['hits']
    return hits
