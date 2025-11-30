-- Save to 'embedding_semantic'
CALL apoc.ml.openai.embedding([text], 'x', {model: 'semantic'}) YIELD embedding
SET n.embedding_semantic = embedding

-- Save to 'my_custom_field'
CALL apoc.ml.openai.embedding([text], 'x', {model: 'semantic'}) YIELD embedding
SET n.my_custom_field = embedding

-- Save to multiple fields with different lenses
MATCH (n:Class {name: 'PaymentService'})
CALL apoc.ml.openai.embedding([n.sourceCode], 'x', {model: 'semantic'}) YIELD embedding AS sem
SET n.emb_semantic = sem
WITH n
CALL apoc.ml.openai.embedding([n.sourceCode], 'x', {model: 'structural'}) YIELD embedding AS str
SET n.emb_structural = str
WITH n
CALL apoc.ml.openai.embedding([n.sourceCode], 'x', {model: 'behavioral'}) YIELD embedding AS beh
SET n.emb_behavioral = beh
RETURN n.name