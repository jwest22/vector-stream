# Vector Stream - Event Embedding and Search

This repository contains a script to generate multi-modal vector embeddings for event descriptions, geolocation data, and timestamps, and to perform similarity searches based on these embeddings using the FAISS library. The script is implemented using PyTorch and the Hugging Face Transformers library.

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

Location Search:
```
Events in New York City:
Distance: 0.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 0.0, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
Distance: 17.02941131591797, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
```

Timestamp search
```
Events with similar timestamps (1625184000):
Distance: 0.0, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
Distance: 350012702720.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 350153441280.0, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
```

Description search
```
Events with similar description ('Event description 1'):
Distance: 0.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 4.750166893005371, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
Distance: 7.550570487976074, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
```

Multi-modal combined index search
```
Combined query results ('New York City', 1625184000, 'Event Description 1'):
Distance: 0.0, Event: {'description': 'Event description 1', 'location': 'New York City', 'timestamp': 1625097600}
Distance: 29795794944.0, Event: {'description': 'Event description 2', 'location': 'Los Angeles', 'timestamp': 1625184000}
Distance: 119198662656.0, Event: {'description': 'Event description 3', 'location': 'New York City', 'timestamp': 1625270400}
```
