Unified Evaluation Workflow Support
====================================

This document maps PyKEEN's capabilities to the Unified Evaluation Workflow, a standardized framework for evaluation harnesses across different modalities.

Strategies are classified into three categories:

- **Natively Supported**: Available immediately after installing PyKEEN (``pip install pykeen``), requires only import statements and minimal configuration (≤2 lines), no external dependencies beyond PyKEEN itself, and no custom implementation or glue code required.

- **Supported via Third-Party Integration**: Requires installing one or more external packages (e.g., ``pykeen[transformers]``, ``pykeen[wandb]``), may require glue code (typically ≤10 lines), has documented integration patterns or official examples, and functionality is enabled through third-party tools in combination with PyKEEN.

- **Not Supported**: Functionality is not available through PyKEEN, either natively or via documented third-party integration.

Phase 0: Provisioning (The Runtime)
-----------------------------------

Step A: Harness Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: PyPI Packages** ✅ **NATIVELY SUPPORTED**

PyKEEN can be installed directly from PyPI using pip:

.. code-block:: bash

    pip install pykeen

Additionally, PyKEEN supports git-based installations:

.. code-block:: bash

    pip install git+https://github.com/pykeen/pykeen.git

*Reference:* See :doc:`installation`

**Strategy 2: Git Clone** ✅ **NATIVELY SUPPORTED**

PyKEEN supports manual installation from source code:

.. code-block:: bash

    git clone https://github.com/pykeen/pykeen.git
    cd pykeen
    pip install -e .

*Reference:* See :doc:`installation`

**Strategy 3: Container Images** ❌ **NOT SUPPORTED**

PyKEEN does not provide prebuilt Docker or OCI container images as part of its native distribution.

**Strategy 4: Binary Packages** ❌ **NOT SUPPORTED**

PyKEEN is a Python package and does not distribute standalone executable binaries.

**Strategy 5: Node Package** ❌ **NOT SUPPORTED**

PyKEEN is a Python package and is not available through Node.js package managers.

Step B: Service Authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Evaluation Platform Authentication** ❌ **NOT SUPPORTED**

PyKEEN does not provide native authentication flows for evaluation platform services or leaderboard submission APIs. While results can be tracked using third-party services, authentication is handled through those external services.

**Strategy 2: API Provider Authentication** ❌ **NOT SUPPORTED**

PyKEEN focuses on local knowledge graph embedding model training and evaluation. It does not natively provide API authentication for commercial model providers or remote inference services.

**Strategy 3: Repository Authentication** 🔌 **SUPPORTED VIA THIRD-PARTY INTEGRATION**

PyKEEN supports token-based authentication with HuggingFace for accessing transformer models through its ``transformers`` extra:

.. code-block:: bash

    pip install pykeen[transformers]

This enables authentication for accessing models and datasets from HuggingFace repositories. This requires the optional ``transformers`` dependency and integration with HuggingFace's authentication system.

*Reference:* The :class:`pykeen.nn.TextRepresentation` class uses ``transformers.AutoModel.from_pretrained`` which respects HuggingFace authentication tokens.

Phase I: Specification (The Contract)
-------------------------------------

Step A: SUT Preparation
~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Model-as-a-Service (Remote Inference)** ❌ **NOT SUPPORTED**

PyKEEN is designed for local training and evaluation of knowledge graph embedding models. It does not natively support HTTP endpoints, SDK clients, or API wrappers for remotely hosted models.

**Strategy 2: Model-in-Process (Local Inference)** ✅ **NATIVELY SUPPORTED**

PyKEEN's core functionality is loading and training knowledge graph embedding models locally in memory:

.. code-block:: python

    from pykeen.pipeline import pipeline
    
    result = pipeline(
        model='TransE',
        dataset='Nations',
    )
    
    # The model is loaded in-process for training and evaluation
    model = result.model

PyKEEN supports 40+ knowledge graph embedding models including TransE, DistMult, ComplEx, RotatE, and more. Models can be loaded from checkpoints and have direct access to internal representations and embeddings.

*Reference:* See :mod:`pykeen.models` for all available models

**Strategy 3: Algorithm Implementation (In-Memory Structures)** ✅ **NATIVELY SUPPORTED**

PyKEEN implements knowledge graph embedding algorithms as in-memory computational procedures. The framework provides:

- Entity and relation embedding representations (:mod:`pykeen.nn.representation`)
- Interaction functions for scoring triples (:mod:`pykeen.nn.modules`)
- Various algorithmic approaches to knowledge graph embeddings

These are instantiated as PyTorch modules without requiring pre-trained weights, though they can be trained and saved.

*Reference:* See :mod:`pykeen.nn` for neural network components

**Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** ❌ **NOT SUPPORTED**

PyKEEN is focused on knowledge graph embeddings and does not support reinforcement learning policies, autonomous agents, or interactive sequential decision-making systems.

Step B: Benchmark Preparation (Inputs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Benchmark Dataset Preparation (Offline)** ✅ **NATIVELY SUPPORTED**

PyKEEN provides 37 built-in datasets for knowledge graph embedding evaluation, including:

- FB15k, FB15k-237
- WN18, WN18RR
- YAGO3-10
- Nations, Kinships
- Hetionet, OpenBioLink, BioKG
- OGB BioKG, OGB WikiKG2
- And many more biomedical and general-purpose knowledge graphs

Datasets are automatically downloaded, cached, and split into train/validation/test partitions:

.. code-block:: python

    from pykeen.datasets import get_dataset
    
    dataset = get_dataset(dataset='FB15k237')
    training = dataset.training
    validation = dataset.validation
    testing = dataset.testing

Additionally, PyKEEN supports loading custom datasets from various formats.

*Reference:* See :mod:`pykeen.datasets` and :doc:`byo/data`

**Strategy 2: Synthetic Data Generation (Generative)** ❌ **NOT SUPPORTED**

PyKEEN does not natively provide synthetic data generation capabilities for creating test data through perturbation or augmentation.

**Strategy 3: Simulation Environment Setup (Simulated)** ❌ **NOT SUPPORTED**

PyKEEN operates on static knowledge graph data and does not support interactive simulation environments.

**Strategy 4: Production Traffic Sampling (Online)** ❌ **NOT SUPPORTED**

PyKEEN is designed for offline evaluation of knowledge graph embeddings and does not natively support sampling or processing real-time production traffic.

Step C: Benchmark Preparation (References)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Judge Preparation** ❌ **NOT SUPPORTED**

PyKEEN uses deterministic and embedding-based metrics for evaluation rather than learned judge models. It does not natively support training or loading specialized judge models for subjective evaluation.

**Strategy 2: Ground Truth Preparation** ✅ **NATIVELY SUPPORTED**

PyKEEN datasets include ground truth annotations in the form of knowledge graph triples. The evaluation framework:

- Pre-loads test triples as ground truth
- Supports filtered evaluation (excluding known true triples from negative samples)
- Maintains entity and relation mappings

.. code-block:: python

    # Ground truth is prepared as part of the dataset
    dataset = get_dataset(dataset='Nations')
    test_triples = dataset.testing.mapped_triples
    
    # Used during evaluation for ranking
    evaluator = RankBasedEvaluator()
    results = evaluator.evaluate(
        model=model,
        mapped_triples=test_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )

*Reference:* See :mod:`pykeen.evaluation` and :doc:`tutorial/understanding_evaluation`

Phase II: Execution (The Run)
-----------------------------

Step A: SUT Invocation
~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Batch Inference** ✅ **NATIVELY SUPPORTED**

PyKEEN's core evaluation workflow executes batch inference over test triples:

.. code-block:: python

    from pykeen.pipeline import pipeline
    
    # Train and evaluate with batch inference
    result = pipeline(
        model='TransE',
        dataset='Nations',
        training_kwargs=dict(num_epochs=100),
    )
    
    # Results contain evaluation metrics from batch inference
    metrics = result.metric_results

The evaluation process scores all entity candidates for head/tail prediction tasks across the entire test set.

*Reference:* See :mod:`pykeen.pipeline` and :mod:`pykeen.evaluation`

**Strategy 2: Interactive Loop** ❌ **NOT SUPPORTED**

PyKEEN evaluates static knowledge graph snapshots and does not support stateful, interactive evaluation loops with sequential actions.

**Strategy 3: Arena Battle** ❌ **NOT SUPPORTED**

PyKEEN evaluates models individually rather than through pairwise comparison or arena-style battles between multiple models on the same inputs.

**Strategy 4: Production Streaming** ❌ **NOT SUPPORTED**

PyKEEN does not natively support continuous processing of live production traffic or real-time metric collection.

Phase III: Assessment (The Score)
---------------------------------

Step A: Individual Scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Deterministic Measurement** ✅ **NATIVELY SUPPORTED**

PyKEEN implements extensive rank-based evaluation metrics that use deterministic calculations:

- Mean Rank (MR)
- Mean Reciprocal Rank (MRR)
- Hits@K (for various K values)
- Adjusted metrics (AAMR, AAMRI, etc.)
- And 44 total ranking and classification metrics

.. code-block:: python

    from pykeen.evaluation import RankBasedEvaluator
    
    evaluator = RankBasedEvaluator()
    results = evaluator.evaluate(
        model=model,
        mapped_triples=test_triples,
    )
    
    # Access deterministic metrics
    mrr = results.get_metric('mean_reciprocal_rank')
    hits_at_10 = results.get_metric('hits_at_10')

*Reference:* See :mod:`pykeen.metrics` and :doc:`tutorial/understanding_evaluation`

**Strategy 2: Embedding Measurement** ✅ **NATIVELY SUPPORTED**

As a knowledge graph embedding framework, PyKEEN's core functionality involves embedding-based measurements. All models:

- Transform entities and relations into learned embedding spaces
- Compute similarity and plausibility scores based on embeddings
- Support various interaction functions (TransE, DistMult, ComplEx, etc.) that operate in embedding space

The entire evaluation process is based on ranking entities according to embedding-based scores.

*Reference:* See :mod:`pykeen.models` and :mod:`pykeen.nn.modules`

**Strategy 3: Subjective Measurement** ❌ **NOT SUPPORTED**

PyKEEN uses deterministic ranking metrics and does not employ LLMs or classifiers as subjective evaluators.

**Strategy 4: Performance Measurement** 🔌 **SUPPORTED VIA THIRD-PARTY INTEGRATION**

PyKEEN natively tracks execution time during training and evaluation:

.. code-block:: python

    result = pipeline(model='TransE', dataset='Nations')
    
    # Training duration is natively tracked
    training_seconds = result.train_seconds
    evaluation_seconds = result.evaluate_seconds

For comprehensive resource monitoring (memory, FLOPs, power consumption, carbon footprint), PyKEEN can be integrated with third-party monitoring tools such as PyTorch profiler, memory_profiler, or carbon tracking libraries, though these integrations require custom implementation.

*Reference:* Training time is reported in pipeline results

Step B: Collective Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Score Aggregation** ✅ **NATIVELY SUPPORTED**

PyKEEN aggregates per-instance ranks into benchmark-level metrics through:

- Arithmetic and geometric means
- Harmonic means (for MRR)
- Quantiles and percentiles
- Stratified metrics (by side, relation type, etc.)

.. code-block:: python

    # Evaluation automatically aggregates individual ranks
    results = evaluator.evaluate(model=model, mapped_triples=test_triples)
    
    # Access aggregated metrics
    overall_mrr = results.get_metric('both.realistic.inverse_harmonic_mean_rank')
    head_mrr = results.get_metric('head.realistic.inverse_harmonic_mean_rank')
    tail_mrr = results.get_metric('tail.realistic.inverse_harmonic_mean_rank')

*Reference:* See :class:`pykeen.evaluation.RankBasedMetricResults`

**Strategy 2: Uncertainty Quantification** ❌ **NOT SUPPORTED**

PyKEEN reports point estimates for metrics but does not natively provide uncertainty quantification through bootstrap resampling or confidence intervals.

Phase IV: Reporting (The Output)
--------------------------------

Step A: Insight Presentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Execution Tracing** ❌ **NOT SUPPORTED**

PyKEEN does not provide detailed step-by-step execution logs showing intermediate reasoning states or decision paths, as it evaluates knowledge graph embedding models rather than tool-using agents.

**Strategy 2: Subgroup Analysis** ✅ **NATIVELY SUPPORTED**

PyKEEN natively provides metrics stratified by:

- Prediction side (head vs. tail)
- Ranking types (optimistic, pessimistic, realistic)

.. code-block:: python

    # Results are automatically stratified by side
    results = evaluator.evaluate(model=model, mapped_triples=test_triples)
    
    # Separate metrics for head and tail prediction
    head_mrr = results.get_metric('head.realistic.inverse_harmonic_mean_rank')
    tail_mrr = results.get_metric('tail.realistic.inverse_harmonic_mean_rank')

For additional stratification (by relation type, entity degree, or custom dimensions), custom analysis code is required.

*Reference:* See :doc:`tutorial/understanding_evaluation`

**Strategy 3: Chart Generation** 🔌 **SUPPORTED VIA THIRD-PARTY INTEGRATION**

PyKEEN supports chart generation through the ``plotting`` extra:

.. code-block:: bash

    pip install pykeen[plotting]

This enables visualization of training losses and basic metric plots using matplotlib and seaborn. Advanced chart generation (radar charts, drift histograms, performance trends) can be implemented using these plotting libraries with PyKEEN's evaluation results.

*Reference:* Install with ``pip install pykeen[plotting]`` for matplotlib and seaborn support

**Strategy 4: Dashboard Creation** 🔌 **SUPPORTED VIA THIRD-PARTY INTEGRATION**

PyKEEN supports integration with external dashboard and tracking services through optional dependencies:

- **MLflow** (``pip install pykeen[mlflow]``)
- **Weights & Biases** (``pip install pykeen[wandb]``)
- **Neptune** (``pip install pykeen[neptune]``)
- **TensorBoard** (``pip install pykeen[tensorboard]``)

.. code-block:: python

    from pykeen.pipeline import pipeline
    
    result = pipeline(
        model='TransE',
        dataset='Nations',
        result_tracker='wandb',  # or 'mlflow', 'neptune', 'tensorboard'
    )

These trackers enable dashboard creation through third-party platforms and require their respective optional dependencies.

*Reference:* See :mod:`pykeen.trackers`

**Strategy 5: Leaderboard Publication** ❌ **NOT SUPPORTED**

PyKEEN does not natively provide functionality for submitting evaluation results to public or private leaderboards. Results must be manually submitted to external leaderboard services.

**Strategy 6: Regression Alerting** ❌ **NOT SUPPORTED**

PyKEEN does not provide automated regression detection or alerting capabilities for comparing results against historical baselines.

Summary
-------

PyKEEN supports the following strategies in the Unified Evaluation Workflow:

**Natively Supported (✅):**

* Phase 0-A-1: PyPI Packages
* Phase 0-A-2: Git Clone
* Phase I-A-2: Model-in-Process (Local Inference)
* Phase I-A-3: Algorithm Implementation (In-Memory Structures)
* Phase I-B-1: Benchmark Dataset Preparation (Offline)
* Phase I-C-2: Ground Truth Preparation
* Phase II-A-1: Batch Inference
* Phase III-A-1: Deterministic Measurement
* Phase III-A-2: Embedding Measurement
* Phase III-B-1: Score Aggregation
* Phase IV-A-2: Subgroup Analysis

**Supported via Third-Party Integration (🔌):**

* Phase 0-B-3: Repository Authentication (requires transformers extra for HuggingFace)
* Phase III-A-4: Performance Measurement (time natively tracked; comprehensive monitoring via third-party tools)
* Phase IV-A-3: Chart Generation (requires plotting extra: matplotlib, seaborn)
* Phase IV-A-4: Dashboard Creation (requires tracker extras: MLflow, W&B, Neptune, TensorBoard)

**Not Supported (❌):**

* Phase 0-A-3: Container Images
* Phase 0-A-4: Binary Packages
* Phase 0-A-5: Node Package
* Phase 0-B-1: Evaluation Platform Authentication
* Phase 0-B-2: API Provider Authentication
* Phase I-A-1: Model-as-a-Service (Remote Inference)
* Phase I-A-4: Policy/Agent Instantiation (Stateful Controllers)
* Phase I-B-2: Synthetic Data Generation (Generative)
* Phase I-B-3: Simulation Environment Setup (Simulated)
* Phase I-B-4: Production Traffic Sampling (Online)
* Phase I-C-1: Judge Preparation
* Phase II-A-2: Interactive Loop
* Phase II-A-3: Arena Battle
* Phase II-A-4: Production Streaming
* Phase III-A-3: Subjective Measurement
* Phase III-B-2: Uncertainty Quantification
* Phase IV-A-1: Execution Tracing
* Phase IV-A-5: Leaderboard Publication
* Phase IV-A-6: Regression Alerting

This analysis reflects PyKEEN's design as a knowledge graph embedding framework focused on offline training and evaluation of embedding models with built-in datasets and deterministic metrics.
