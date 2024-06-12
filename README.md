# Vector Stream
This repository provides a script to generate multi-modal vector embeddings for event descriptions, geolocation data, and timestamps. It uses PyTorch and Hugging Face Transformers, with FAISS for similarity searches.

## Multi-Modal Vector Embeddings and Similarity Search

### Embedding Details
Description: 768 dimensions using BERT
Location: 768 dimensions using BERT
Timestamp: 128 dimensions using a linear layer
Total Dimensionality: 1664 dimensions
These embeddings are combined and can be transformed into a shared latent space.

### Multi-Modal Embedding Approach
**Pros:**
Granular Control: Detailed analysis of each component.
Integrated Context: Learns dependencies between different data types.
Flexibility: Queries can target any subset of modalities.

**Cons:**
Complex Training: Needs careful design and tuning.
Increased Model Complexity: Requires sophisticated architecture.

### Traditional Single Index Approach
Combines all values into a single text field before embedding generation.

**Pros:**
Simplicity: Easier to implement with a single model.
Unified Context: Embedding captures integrated context.

**Cons:**
Loss of Specificity: Less distinct representation of data types.
Potential for Overload: Concatenated input may reduce learning effectiveness.
Flexibility Limitations: Less flexible for independent queries.

## Table of Contents
- [Requirements](#requirements)
- [Examples](#examples)

## Requirements
- Python 3.7+
- PyTorch
- transformers
- faiss
- numpy

You can install the required libraries using pip:
```bash
pip install torch transformers faiss-cpu numpy
```

## Examples

**Location Search:**
```
Events in New York City:
Distance: 0.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 0.0, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
Distance: 17.02941131591797, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
```

**Timestamp search**
```
Events with similar timestamps (1625184000):
Distance: 0.0, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
Distance: 350012702720.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 350153441280.0, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
```

**Description search**
```
Events with similar description ('Event description 1'):
Distance: 0.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 4.750166893005371, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
Distance: 7.550570487976074, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
```

**Multi-modal combined index search**
```
Combined query results ('New York City', 1625184000, 'Event Description 1'):
Distance: 0.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 29795794944.0, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
Distance: 119198662656.0, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
```
