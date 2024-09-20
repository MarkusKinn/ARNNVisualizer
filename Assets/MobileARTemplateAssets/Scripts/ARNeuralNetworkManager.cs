using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class ARNeuralNetworkManager : MonoBehaviour
{
    public GameObject neuronPrefab;
    public Button addLayerButton;
    public Button removeLayerButton;
    public TMP_InputField neuronsInputField;
    public float neuronSize = 0.2f;
    public float heightOffset = 0.2f;
    public Material connectionMaterial;

    [Header("Training Settings")]
    public Button trainButton;
    public TMP_InputField epochsInputField;
    public TMP_InputField learningRateInputField;
    public TMP_InputField batchSizeInputField;
    public int trainingDataPoints = 100;
    public float minX = -10f;
    public float maxX = 10f;

    private NeuralNetwork neuralNetwork;

    [Header("Layout Settings")]
    public float minLayerDistance = 0.3f;
    public float maxLayerDistance = 1f;
    public float minNeuronDistance = 0.2f;
    public float maxNeuronDistance = 0.6f;

    [Header("Data Flow Visualization")]
    public Button simulateDataFlowButton;
    public float dataFlowSpeed = 1f;
    public Color lowValueColor = Color.blue;
    public Color highValueColor = Color.red;

    [Header("Optimization Settings")]
    public int maxNeuronsPerLayer = 100;
    public int maxConnectionsPerLayer = 1000;

    [Header("Visualization Settings")]
    public float visualizationScale = 0.1f;
    public float sphereSize = 0.05f;

    [Header("User Input")]
    public TMP_InputField userInputField;
    public Button predictButton;
    public TextMeshProUGUI predictionResultText;


    private List<int> networkStructure = new List<int>();
    private List<List<GameObject>> layerObjects = new List<List<GameObject>>();
    private List<LineRenderer> connectionObjects = new List<LineRenderer>();
    private List<List<float>> layerValues = new List<List<float>>();
    private List<List<Renderer>> neuronRenderers = new List<List<Renderer>>();

    private GameObject networkContainer;
    private List<GameObject> neuronPool = new List<GameObject>();
    private List<LineRenderer> connectionPool = new List<LineRenderer>();

    private float minY;
    private float maxY;

    void Start()
    {
        InitializeUI();
        CreateNetworkContainer();
        InitializeObjectPools();
    }

    void InitializeObjectPools()
    {
        // Initialize neuron pool
        for (int i = 0; i < maxNeuronsPerLayer; i++)
        {
            GameObject neuron = Instantiate(neuronPrefab, Vector3.zero, Quaternion.identity, networkContainer.transform);
            neuron.SetActive(false);
            neuronPool.Add(neuron);
        }

        // Initialize connection pool
        for (int i = 0; i < maxConnectionsPerLayer; i++)
        {
            GameObject connectionObj = new GameObject("Connection");
            connectionObj.transform.SetParent(networkContainer.transform);
            LineRenderer lineRenderer = connectionObj.AddComponent<LineRenderer>();
            lineRenderer.material = new Material(connectionMaterial);
            lineRenderer.startWidth = 0.01f;
            lineRenderer.endWidth = 0.01f;
            lineRenderer.positionCount = 2;
            lineRenderer.useWorldSpace = true;
            connectionObj.SetActive(false);
            connectionPool.Add(lineRenderer);
        }
    }
    GameObject GetNeuronFromPool()
    {
        foreach (var neuron in neuronPool)
        {
            if (!neuron.activeSelf)
            {
                neuron.SetActive(true);
                return neuron;
            }
        }
        Debug.LogWarning("Neuron pool exhausted. Consider increasing pool size.");
        return null;
    }

    LineRenderer GetConnectionFromPool()
    {
        foreach (var connection in connectionPool)
        {
            if (!connection.gameObject.activeSelf)
            {
                connection.gameObject.SetActive(true);
                return connection;
            }
        }
        Debug.LogWarning("Connection pool exhausted. Consider increasing pool size.");
        return null;
    }


    void CreateNetworkContainer()
    {
        networkContainer = new GameObject("Neural Network");
        networkContainer.transform.SetParent(transform);
    }

    void InitializeUI()
    {
        addLayerButton.onClick.AddListener(AddLayer);
        removeLayerButton.onClick.AddListener(RemoveLayer);
        simulateDataFlowButton.onClick.AddListener(SimulateDataFlow);

        neuronsInputField.contentType = TMP_InputField.ContentType.IntegerNumber;
        neuronsInputField.characterValidation = TMP_InputField.CharacterValidation.Integer;
        neuronsInputField.characterLimit = 3;

        simulateDataFlowButton.interactable = false;

        trainButton.onClick.AddListener(TrainNetwork);
        epochsInputField.contentType = TMP_InputField.ContentType.IntegerNumber;
        epochsInputField.characterValidation = TMP_InputField.CharacterValidation.Integer;
        learningRateInputField.contentType = TMP_InputField.ContentType.DecimalNumber;
        learningRateInputField.characterValidation = TMP_InputField.CharacterValidation.Decimal;

        trainButton.interactable = false;

        batchSizeInputField.contentType = TMP_InputField.ContentType.IntegerNumber;
        batchSizeInputField.characterValidation = TMP_InputField.CharacterValidation.Integer;

        userInputField.contentType = TMP_InputField.ContentType.DecimalNumber;
        predictButton.onClick.AddListener(PredictUserInput);
    }

    List<float> GenerateRandomInputData()
    {
        List<float> inputData = new List<float>();
        for (int i = 0; i < networkStructure[0]; i++)
        {
            inputData.Add(Random.value);
        }
        return inputData;
    }

    public void SimulateDataFlow()
    {
        if (networkStructure.Count == 0)
        {
            Debug.LogWarning("No layers in the network. Please add layers first.");
            return;
        }

        List<float> inputData = GenerateRandomInputData();
        StartCoroutine(AnimateDataFlow(inputData));
    }

    IEnumerator AnimateDataFlow(List<float> inputData)
    {
        Debug.Log("Starting AnimateDataFlow coroutine");
        // Set input layer values
        layerValues[0] = inputData;

        // Simulate data flow through the network
        for (int layerIndex = 0; layerIndex < networkStructure.Count; layerIndex++)
        {
            Debug.Log($"Processing layer {layerIndex}");
            // Update neuron colors
            UpdateNeuronColors(layerIndex);

            // If not the last layer, calculate next layer values
            if (layerIndex < networkStructure.Count - 1)
            {
                layerValues[layerIndex + 1] = SimulateLayerOutput(layerValues[layerIndex], networkStructure[layerIndex + 1]);
            }

            // Animate connections
            if (layerIndex > 0)
            {
                AnimateConnections(layerIndex - 1, layerIndex);
            }

            yield return new WaitForSeconds(dataFlowSpeed);
        }

        Debug.Log("Animation complete");
        // Reset colors after animation
        yield return new WaitForSeconds(dataFlowSpeed);
        ResetColors();
    }
    void UpdateNeuronColors(int layerIndex)
    {
        if (layerIndex >= layerValues.Count || layerIndex >= neuronRenderers.Count)
        {
            Debug.LogError($"Invalid layer index: {layerIndex}");
            return;
        }

        float minValue = layerValues[layerIndex].Min();
        float maxValue = layerValues[layerIndex].Max();

        for (int i = 0; i < layerValues[layerIndex].Count; i++)
        {
            if (i >= neuronRenderers[layerIndex].Count)
            {
                Debug.LogWarning($"Neuron index out of range: {i} in layer {layerIndex}");
                continue;
            }

            float normalizedValue = Mathf.InverseLerp(minValue, maxValue, layerValues[layerIndex][i]);
            Color neuronColor = Color.Lerp(lowValueColor, highValueColor, normalizedValue);

            Renderer renderer = neuronRenderers[layerIndex][i];
            if (renderer != null && renderer.material != null)
            {
                // Change this line:
                renderer.material.color = neuronColor;
                // To this:
                renderer.material.SetColor("_Color", neuronColor);
                Debug.Log($"Set color for neuron {i} in layer {layerIndex} to {neuronColor}");
            }
            else
            {
                Debug.LogWarning($"Cannot update color for neuron {i} in layer {layerIndex}");
            }
        }
    }
    List<float> SimulateLayerOutput(List<float> inputValues, int outputNeurons)
    {
        // Simple simulation: random weights and sigmoid activation
        List<float> outputValues = new List<float>();
        for (int i = 0; i < outputNeurons; i++)
        {
            float sum = 0;
            foreach (float input in inputValues)
            {
                sum += input * Random.value; // Random weight
            }
            outputValues.Add(Sigmoid(sum));
        }
        return outputValues;
    }

    float Sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

    void AnimateConnections(int fromLayer, int toLayer)
    {
        int connectionIndex = 0;
        for (int i = 0; i < layerObjects[fromLayer].Count; i++)
        {
            for (int j = 0; j < layerObjects[toLayer].Count; j++)
            {
                if (connectionIndex < connectionObjects.Count)
                {
                    float fromValue = layerValues[fromLayer][i];
                    float toValue = layerValues[toLayer][j];
                    float averageValue = (fromValue + toValue) / 2f;

                    Color startColor = Color.Lerp(lowValueColor, highValueColor, fromValue);
                    Color endColor = Color.Lerp(lowValueColor, highValueColor, toValue);
                    Color averageColor = Color.Lerp(lowValueColor, highValueColor, averageValue);

                    LineRenderer lineRenderer = connectionObjects[connectionIndex];
                    lineRenderer.startColor = startColor;
                    lineRenderer.endColor = endColor;

                    // Set the color of the line renderer's material
                    lineRenderer.material.color = averageColor;

                    Debug.Log($"Set colors for connection {connectionIndex} - Start: {startColor}, End: {endColor}, Average: {averageColor}");
                    connectionIndex++;
                }
                else
                {
                    Debug.LogWarning("Not enough connection objects");
                    return;
                }
            }
        }
    }

    void ResetColors()
    {
        foreach (var layerRenderers in neuronRenderers)
        {
            foreach (var renderer in layerRenderers)
            {
                if (renderer != null && renderer.material != null)
                {
                    renderer.material.color = Color.white;
                }
            }
        }

        foreach (var connection in connectionObjects)
        {
            connection.startColor = Color.white;
            connection.endColor = Color.white;
            connection.material.color = Color.white;
        }
    }

    public void AddLayer()
    {
        int neurons;
        if (int.TryParse(neuronsInputField.text, out neurons) && neurons > 0)
        {
            networkStructure.Add(neurons);
        }
        else
        {
            neurons = 10; // Default value if input is invalid
            networkStructure.Add(neurons);
        }
        VisualizeNetwork();
        Debug.Log($"Added layer with {neurons} neurons");
        neuronsInputField.text = "";

        simulateDataFlowButton.interactable = networkStructure.Count > 0;

        trainButton.interactable = networkStructure.Count > 1;

    }
    public void RemoveLayer()
    {
        if (networkStructure.Count > 0)
        {
            networkStructure.RemoveAt(networkStructure.Count - 1);
            VisualizeNetwork();
            Debug.Log("Removed last layer");
        }
        else
        {
            Debug.LogWarning("No layers to remove");
        }

        simulateDataFlowButton.interactable = networkStructure.Count > 0;

        trainButton.interactable = networkStructure.Count > 1;

    }

    void VisualizeNetwork()
    {
        ClearVisualization();

        if (networkStructure.Count == 0)
        {
            return; // No layers to visualize
        }

        int maxNeuronsInLayer = Mathf.Min(networkStructure.Max(), maxNeuronsPerLayer);
        float layerDistance = CalculateLayerDistance();

        for (int layerIndex = 0; layerIndex < networkStructure.Count; layerIndex++)
        {
            List<GameObject> layerNeurons = new List<GameObject>();
            List<Renderer> layerRenderers = new List<Renderer>();
            int neuronsInLayer = Mathf.Min(networkStructure[layerIndex], maxNeuronsPerLayer);
            float neuronDistance = CalculateNeuronDistance(neuronsInLayer, maxNeuronsInLayer);

            for (int neuronIndex = 0; neuronIndex < neuronsInLayer; neuronIndex++)
            {
                Vector3 position = CalculateNeuronPosition(layerIndex, neuronIndex, neuronsInLayer, layerDistance, neuronDistance);
                GameObject neuron = GetNeuronFromPool();
                if (neuron != null)
                {
                    neuron.transform.position = position;
                    neuron.transform.localScale = Vector3.one * neuronSize;
                    layerNeurons.Add(neuron);

                    Renderer renderer = neuron.GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        layerRenderers.Add(renderer);
                    }
                }
            }

            layerObjects.Add(layerNeurons);
            neuronRenderers.Add(layerRenderers);

            if (layerIndex > 0)
            {
                CreateConnections(layerObjects[layerIndex - 1], layerNeurons);
            }
        }

        // Initialize layerValues
        layerValues = new List<List<float>>();
        for (int i = 0; i < networkStructure.Count; i++)
        {
            layerValues.Add(new List<float>(new float[networkStructure[i]]));
        }
    }
    float CalculateLayerDistance()
    {
        float t = Mathf.InverseLerp(2, 10, networkStructure.Count);
        return Mathf.Lerp(maxLayerDistance, minLayerDistance, t);
    }

    float CalculateNeuronDistance(int neuronsInLayer, int maxNeuronsInLayer)
    {
        float t = Mathf.InverseLerp(1, maxNeuronsInLayer, neuronsInLayer);
        return Mathf.Lerp(maxNeuronDistance, minNeuronDistance, t);
    }

    Vector3 CalculateNeuronPosition(int layerIndex, int neuronIndex, int neuronsInLayer, float layerDistance, float neuronDistance)
    {
        float x = layerIndex * layerDistance;
        float y = (neuronIndex - (neuronsInLayer - 1) / 2f) * neuronDistance;
        return new Vector3(x, y + heightOffset, 0);
    }

    void CreateConnections(List<GameObject> fromLayer, List<GameObject> toLayer)
    {
        int connectionCount = 0;
        foreach (GameObject fromNeuron in fromLayer)
        {
            foreach (GameObject toNeuron in toLayer)
            {
                if (connectionCount >= maxConnectionsPerLayer)
                {
                    Debug.LogWarning("Max connections per layer reached. Some connections will not be visualized.");
                    return;
                }

                LineRenderer lineRenderer = GetConnectionFromPool();
                if (lineRenderer != null)
                {
                    ConnectionUpdater updater = lineRenderer.gameObject.GetComponent<ConnectionUpdater>();
                    if (updater == null)
                    {
                        updater = lineRenderer.gameObject.AddComponent<ConnectionUpdater>();
                    }
                    updater.SetNeurons(fromNeuron.transform, toNeuron.transform);
                    updater.lineRenderer = lineRenderer;

                    connectionObjects.Add(lineRenderer);
                    connectionCount++;
                }
            }
        }
    }
    void ClearVisualization()
    {
        foreach (var layer in layerObjects)
        {
            foreach (var neuron in layer)
            {
                neuron.SetActive(false);
            }
        }
        layerObjects.Clear();

        foreach (var connection in connectionObjects)
        {
            connection.gameObject.SetActive(false);
        }
        connectionObjects.Clear();

        neuronRenderers.Clear();
        layerValues.Clear();
    }

    void TrainNetwork()
    {
        if (networkStructure.Count < 2)
        {
            Debug.LogWarning("Network should have at least two layers (input and output).");
            return;
        }

        int epochs = 100000; // Increase number of epochs
        float learningRate = 0.01f; // Adjust learning rate
        int batchSize = 32;

        neuralNetwork = new NeuralNetwork(networkStructure);

        List<Vector> inputs = new List<Vector>();
        List<Vector> targets = new List<Vector>();

        // Generate training data for X^3 function
        minY = float.MaxValue;
        maxY = float.MinValue;
        for (int i = 0; i < trainingDataPoints; i++)
        {
            float x = Random.Range(minX, maxX);
            float y = x * x * x;
            minY = Mathf.Min(minY, y);
            maxY = Mathf.Max(maxY, y);

            Vector input = new Vector(1);
            input[0] = x;
            inputs.Add(input);

            Vector target = new Vector(1);
            target[0] = y;
            targets.Add(target);
        }

        // Normalize inputs and targets
        for (int i = 0; i < inputs.Count; i++)
        {
            inputs[i] = neuralNetwork.Normalize(inputs[i], minX, maxX);
            targets[i] = neuralNetwork.Normalize(targets[i], minY, maxY);
        }

        neuralNetwork.TrainMiniBatch(inputs, targets, epochs, learningRate, batchSize);

        // Visualize the trained network
        VisualizeTrainedNetwork();
    }

    void VisualizeTrainedNetwork()
    {
        // Clear existing visualization
        ClearVisualization();

        // Visualize the network structure
        VisualizeNetwork();

        // Visualize the learned function
        StartCoroutine(VisualizeLearnedFunction());
    }

    IEnumerator VisualizeLearnedFunction()
    {
        int numPoints = 100;
        List<Vector3> points = new List<Vector3>();

        for (int i = 0; i < numPoints; i++)
        {
            float x = Mathf.Lerp(minX, maxX, (float)i / (numPoints - 1));
            Vector input = new Vector(1);
            input[0] = x;

            // Normalize the input
            Vector normalizedInput = neuralNetwork.Normalize(input, minX, maxX);

            Vector normalizedOutput = neuralNetwork.FeedForward(normalizedInput);
            Vector output = neuralNetwork.Denormalize(normalizedOutput, minY, maxY);
            float predictedY = output[0];
            float actualY = x * x * x;

            // Scale down the position
            Vector3 position = new Vector3(x, predictedY, 0) * visualizationScale;
            points.Add(position);

            // Visualize the point
            GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            point.transform.position = position;
            point.transform.localScale = Vector3.one * sphereSize;

            // Color the point based on prediction accuracy
            float error = Mathf.Abs(predictedY - actualY);
            float maxError = maxY - minY;
            Color pointColor = Color.Lerp(Color.green, Color.red, error / maxError);
            point.GetComponent<Renderer>().material.color = pointColor;

            yield return null;
        }

        // Create a line through all points
        GameObject lineObject = new GameObject("Prediction Line");
        LineRenderer lineRenderer = lineObject.AddComponent<LineRenderer>();
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = Color.cyan;
        lineRenderer.endColor = Color.cyan;
        lineRenderer.startWidth = 0.02f;
        lineRenderer.endWidth = 0.02f;
        lineRenderer.positionCount = points.Count;
        lineRenderer.SetPositions(points.ToArray());

        // Visualize the actual function
        LineRenderer actualFunctionLine = new GameObject("Actual Function").AddComponent<LineRenderer>();
        actualFunctionLine.material = new Material(Shader.Find("Sprites/Default"));
        actualFunctionLine.startColor = Color.blue;
        actualFunctionLine.endColor = Color.blue;
        actualFunctionLine.startWidth = 0.02f;
        actualFunctionLine.endWidth = 0.02f;
        actualFunctionLine.positionCount = numPoints;

        for (int i = 0; i < numPoints; i++)
        {
            float x = Mathf.Lerp(minX, maxX, (float)i / (numPoints - 1));
            float y = x * x * x;
            Vector3 position = new Vector3(x, y, 0) * visualizationScale;
            actualFunctionLine.SetPosition(i, position);
        }
    }

    void PredictUserInput()
    {
        if (neuralNetwork == null)
        {
            Debug.LogWarning("Neural network is not trained yet. Please train the network first.");
            return;
        }

        float userInput;
        if (float.TryParse(userInputField.text, out userInput))
        {
            Vector input = new Vector(1);
            input[0] = userInput;

            // Normalize the input
            Vector normalizedInput = neuralNetwork.Normalize(input, minX, maxX);

            Vector normalizedOutput = neuralNetwork.FeedForward(normalizedInput);
            Vector output = neuralNetwork.Denormalize(normalizedOutput, minY, maxY);
            float predictedY = output[0];

            // Display the result
            predictionResultText.text = $"Input: {userInput}\nPredicted Output: {predictedY:F2}";
        }
        else
        {
            Debug.LogWarning("Invalid input. Please enter a valid number.");
        }
    }

    public class NeuronMover : MonoBehaviour
    {
        private Vector3 offset;
        private float zCoord;

        void OnMouseDown()
        {
            zCoord = Camera.main.WorldToScreenPoint(gameObject.transform.position).z;
            offset = gameObject.transform.position - GetMouseWorldPos();
        }

        void OnMouseDrag()
        {
            transform.position = GetMouseWorldPos() + offset;
        }

        private Vector3 GetMouseWorldPos()
        {
            Vector3 mousePoint = Input.mousePosition;
            mousePoint.z = zCoord;
            return Camera.main.ScreenToWorldPoint(mousePoint);
        }
    }

    public class ConnectionUpdater : MonoBehaviour
    {
        public LineRenderer lineRenderer;
        private Transform fromNeuron;
        private Transform toNeuron;

        public void SetNeurons(Transform from, Transform to)
        {
            fromNeuron = from;
            toNeuron = to;
        }

        void LateUpdate()
        {
            if (fromNeuron != null && toNeuron != null)
            {
                lineRenderer.SetPosition(0, fromNeuron.position);
                lineRenderer.SetPosition(1, toNeuron.position);
            }
        }
    }
}