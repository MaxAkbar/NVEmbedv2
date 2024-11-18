import base64
import numpy as np
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from sentence_transformers import SentenceTransformer

# Initialize the Flask app
app = Flask(__name__)
swagger = Swagger(app)

# Load the model
model = SentenceTransformer('NV-Embed-v2', trust_remote_code=True)
model.max_seq_length = 32768
model.tokenizer.padding_side = "right"

# Utility function to add EOS token
def add_eos(input_examples):
    input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
    return input_examples

# Define the embedding endpoint
@app.route('/v1/embeddings', methods=['POST'])
@swag_from({
    'tags': ['Embeddings'],
    'parameters': [
        {
            'name': 'input',
            'in': 'body',
            'type': 'string',
            'required': True,
            'description': 'The input text to generate embeddings for.',
            'schema': {
                'type': 'object',
                'properties': {
                    'input': {
                        'type': 'string',
                        'example': 'The food was delicious and the waiter...'
                    },
                    'model': {
                        'type': 'string',
                        'example': 'nvidia/NV-Embed-v2'
                    },
                    'batch_size': {
                        'type': 'integer',
                        'example': 2
                    },
                    'normalize_embeddings': {
                        'type': 'boolean',
                        'example': True
                    }
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': 'The generated embedding response',
            'schema': {
                'type': 'object',
                'properties': {
                    'object': {'type': 'string', 'example': 'list'},
                    'data': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'object': {'type': 'string', 'example': 'embedding'},
                                'embedding': {
                                    'type': 'array',
                                    'items': {'type': 'number'},
                                    'example': [0.0023, -0.0093, 0.0021]  # Limited example array instead of ...
                                },
                                'index': {'type': 'integer', 'example': 0}
                            }
                        }
                    },
                    'model': {'type': 'string', 'example': 'nvidia/NV-Embed-v2'},
                    'usage': {
                        'type': 'object',
                        'properties': {
                            'prompt_tokens': {'type': 'integer', 'example': 8},
                            'total_tokens': {'type': 'integer', 'example': 8}
                        }
                    }
                }
            }
        },
        400: {
            'description': 'Error: Bad request'
        }
    }
})
def embed():
    data = request.json
    if 'input' not in data:
        return jsonify({'error': 'Please provide input text to embed.'}), 400

    texts = data['input']
    if isinstance(texts, str):
        texts = [texts]

    batch_size = data.get('batch_size', 2)
    normalize_embeddings = data.get('normalize_embeddings', True)

    # Get the embeddings
    embeddings = model.encode(add_eos(texts), batch_size=batch_size, normalize_embeddings=normalize_embeddings)

    # Format the response similar to OpenAI API
    response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": base64.b64encode(embedding.astype(np.float32).tobytes()).decode('utf-8'),
                "index": idx
            } for idx, embedding in enumerate(embeddings)
        ],
        "model": "NV-Embed-v2",
        "usage": {
            "prompt_tokens": len(texts),
            "total_tokens": len(texts)
        }
    }

    return jsonify(response)

# Add custom error handler for TypeError
@app.errorhandler(TypeError)
def handle_type_error(error):
    return jsonify({
        "error": "JSON serialization error",
        "message": str(error)
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)