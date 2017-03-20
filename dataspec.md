# Data Spec for Learning to Rank

## Accepted input formats

*Domain-specific formats.*


### Unit-Similarity-Queries

This is to train a new unit similarity model using an existing similarity matrix, potentially with new features.


```
{
	"query_unit": {
		"features": "array"
	},
	"result_units": {
		"type": "array",
		"items": {
			"type": "object",
			"properties": {
				"features": "array",
				"score": "float"
			}
		}
	}
}
```

Example:

```
{
	"query_unit": {"id": "124195", "features": [1, 0, -3]},
	"units": [
		{"id": "124132", "features": [0, 5, -1], "score": 0.39},
		{"id": "934523", "features": [1, 1, 3], "score": 0.62},
		...
	]
}
```

The training algo only uses the `score` field in terms of its ranking, rather than trying to do regression on the score itself. So if you perform monotonic transform on the scores, it will not change the resulting model. The score can come from anywhere. For example, it can be conversion rate; or it can be an existing similarity score.

### Sort

```
{
	"query": {
		"type": "object",
		"properties": {
			"destination_id": "string",
			"filters": "array"
		}
	},
	"result_units": {
		"type": "array",
		"items": {
			"type": "object"
		}
	}
}
```

A result unit can be a general JSON object. It is up to the vectorizer to interpret the data.


*You can also choose to use more generic formats.*

### Triplets-Features

In each data file, each row is a JSON object specified as the following:

```
{
	"feature_dim": {
		"type": "int",
		"example": 50,
		"required": false,
		"description": "Dimensionality of features. Depending on what type of model is used, the feature dimension may be different from row to row. If you specify feature_dim, then the algo will raise an exception if the actual features have a different dimension."
	},
	"query_features": {
		"required": false,
		"type": "array",
		"items": {
			"type": "float"
		}
	},
	"higher_features": {
		"required": true,
		"type": "array",
		"items": {
			"type": "float"
		}
	},
	"lower_features": {
		"required": true,
		"type": "array",
		"items": {
			"type": "float"
		}
	}
}
```

When "query_features" is not provided, the triplets become pairs.

### Triplets-Objects

This is a "free-style" data spec, where you can choose any structure for your input data, as long as your vectorizer and your model can handle them.

```
{
	"query_object": {"required": false},
	"higher_object": {"required": true},
	"lower_object": {"required": true}
}
```

### Elements-Features

This is the typical input format for learning to rank: (X, y, queries).

{
	"feature_dim": {
		"required": false,
		"type": "int"
	}
	"query": {
		"type": "string",
		"required": false
	},
	"features": {
		"type"" "array",
		"items": "float"
	},
	"label": "float"
}

The training algo will generate triplets using label as the ranking score.

### Elements-Objects

This is the freestyle counterpart.

{
	"query": {
		"type": "string",
		"required": false
	},
	"input": "object",
	"output": "object"
}