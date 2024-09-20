using System;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork
{
    private List<int> layers;
    private List<Matrix> weights;
    private List<Vector> biases;
    private List<Vector> activations;
    private List<Vector> zs;

    public NeuralNetwork(List<int> layers)
    {
        this.layers = layers;
        weights = new List<Matrix>();
        biases = new List<Vector>();
        activations = new List<Vector>();
        zs = new List<Vector>();

        // Initialize weights and biases
        for (int i = 1; i < layers.Count; i++)
        {
            weights.Add(new Matrix(layers[i], layers[i - 1], true));
            biases.Add(new Vector(layers[i]));
        }
    }

    public Vector FeedForward(Vector input)
    {
        activations.Clear();
        zs.Clear();
        Vector activation = input;
        activations.Add(activation);

        for (int i = 0; i < weights.Count; i++)
        {
            Vector z = weights[i] * activation + biases[i];
            zs.Add(z);
            activation = i == weights.Count - 1 ? z : z.Apply(ReLU); // Remove activation for output layer
            activations.Add(activation);
        }
        return activation;
    }

    public void TrainMiniBatch(List<Vector> inputs, List<Vector> targets, int epochs, float initialLearningRate, int batchSize)
    {
        float decayRate = 0.95f;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float learningRate = initialLearningRate * Mathf.Pow(decayRate, epoch / 100f);
            float totalLoss = 0;
            for (int i = 0; i < inputs.Count; i += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, inputs.Count - i);
                List<Vector> batchInputs = inputs.GetRange(i, actualBatchSize);
                List<Vector> batchTargets = targets.GetRange(i, actualBatchSize);
                totalLoss += TrainBatch(batchInputs, batchTargets, learningRate);
            }
            totalLoss /= inputs.Count;
            if (epoch % 100 == 0)
            {
                Debug.Log($"Epoch {epoch + 1}, Loss: {totalLoss}");
            }
        }
    }

    private float TrainBatch(List<Vector> inputs, List<Vector> targets, float learningRate)
    {
        List<Matrix> weightGradients = new List<Matrix>();
        List<Vector> biasGradients = new List<Vector>();
        float batchLoss = 0;

        for (int l = 0; l < weights.Count; l++)
        {
            weightGradients.Add(new Matrix(weights[l].Rows, weights[l].Cols));
            biasGradients.Add(new Vector(biases[l].Size));
        }

        for (int i = 0; i < inputs.Count; i++)
        {
            Vector output = FeedForward(inputs[i]);
            batchLoss += MeanSquaredError(output, targets[i]);
            Backpropagation(targets[i], weightGradients, biasGradients);
        }

        for (int l = 0; l < weights.Count; l++)
        {
            weights[l] -= weightGradients[l].Multiply(learningRate / inputs.Count);
            biases[l] -= biasGradients[l].Multiply(learningRate / inputs.Count);
        }

        return batchLoss / inputs.Count;
    }

    private void Backpropagation(Vector target, List<Matrix> weightGradients, List<Vector> biasGradients)
    {
        int numLayers = weights.Count;
        List<Vector> deltas = new List<Vector>();

        // Calculate delta for output layer (no activation function for output layer)
        Vector delta = activations[numLayers] - target;
        deltas.Add(delta);

        // Calculate deltas for hidden layers
        for (int l = numLayers - 1; l > 0; l--)
        {
            delta = (weights[l].Transpose() * delta).Hadamard(zs[l - 1].Apply(ReLUPrime));
            deltas.Insert(0, delta);
        }

        // Update weight and bias gradients
        for (int l = 0; l < numLayers; l++)
        {
            weightGradients[l] += Matrix.OuterProduct(deltas[l], activations[l]);
            biasGradients[l] += deltas[l];
        }
    }

    private float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    private float SigmoidPrime(float x)
    {
        float s = Sigmoid(x);
        return s * (1 - s);
    }

    private float ReLU(float x)
    {
        return Mathf.Max(0, x);
    }

    private float ReLUPrime(float x)
    {
        return x > 0 ? 1 : 0;
    }

    public float MeanSquaredError(Vector output, Vector target)
    {
        float sum = 0;
        for (int i = 0; i < output.Size; i++)
        {
            float diff = output[i] - target[i];
            sum += diff * diff;
        }
        return sum / output.Size;
    }

    public Vector Normalize(Vector input, float min, float max)
    {
        return input.Apply(x => (x - min) / (max - min));
    }

    public Vector Denormalize(Vector input, float min, float max)
    {
        return input.Apply(x => x * (max - min) + min);
    }
}
public class Matrix
{
    private float[,] data;
    public int Rows { get; private set; }
    public int Cols { get; private set; }

    public Matrix(int rows, int cols, bool initialize = false)
    {
        Rows = rows;
        Cols = cols;
        data = new float[rows, cols];

        if (initialize)
        {
            // Xavier/Glorot initialization
            float stdDev = Mathf.Sqrt(2f / (rows + cols));
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i, j] = UnityEngine.Random.Range(-stdDev, stdDev);
        }
    }

    public static Vector operator *(Matrix m, Vector v)
    {
        if (m.Cols != v.Size)
            throw new ArgumentException("Matrix columns must match Vector size.");

        Vector result = new Vector(m.Rows);
        for (int i = 0; i < m.Rows; i++)
        {
            float sum = 0;
            for (int j = 0; j < m.Cols; j++)
            {
                sum += m.data[i, j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    public Matrix Transpose()
    {
        Matrix result = new Matrix(Cols, Rows);
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < Cols; j++)
                result.data[j, i] = data[i, j];
        return result;
    }

    public static Matrix OuterProduct(Vector a, Vector b)
    {
        Matrix result = new Matrix(a.Size, b.Size);
        for (int i = 0; i < a.Size; i++)
            for (int j = 0; j < b.Size; j++)
                result.data[i, j] = a[i] * b[j];
        return result;
    }

    public Matrix Multiply(float scalar)
    {
        Matrix result = new Matrix(Rows, Cols);
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < Cols; j++)
                result.data[i, j] = data[i, j] * scalar;
        return result;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException("Matrices must have the same dimensions.");

        Matrix result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.data[i, j] = a.data[i, j] + b.data[i, j];
        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException("Matrices must have the same dimensions.");

        Matrix result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Cols; j++)
                result.data[i, j] = a.data[i, j] - b.data[i, j];
        return result;
    }
}

public class Vector
{
    private float[] data;
    public int Size { get; private set; }

    public Vector(int size)
    {
        Size = size;
        data = new float[size];
    }

    public float this[int index]
    {
        get { return data[index]; }
        set { data[index] = value; }
    }

    public static Vector operator +(Vector a, Vector b)
    {
        if (a.Size != b.Size)
            throw new ArgumentException("Vectors must have the same size.");

        Vector result = new Vector(a.Size);
        for (int i = 0; i < a.Size; i++)
        {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    public static Vector operator -(Vector a, Vector b)
    {
        if (a.Size != b.Size)
            throw new ArgumentException("Vectors must have the same size.");

        Vector result = new Vector(a.Size);
        for (int i = 0; i < a.Size; i++)
        {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    public Vector Hadamard(Vector other)
    {
        if (Size != other.Size)
            throw new ArgumentException("Vectors must have the same size.");

        Vector result = new Vector(Size);
        for (int i = 0; i < Size; i++)
        {
            result[i] = data[i] * other[i];
        }
        return result;
    }

    public Vector Multiply(float scalar)
    {
        Vector result = new Vector(Size);
        for (int i = 0; i < Size; i++)
        {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    public Vector Apply(Func<float, float> function)
    {
        Vector result = new Vector(Size);
        for (int i = 0; i < Size; i++)
        {
            result[i] = function(data[i]);
        }
        return result;
    }

    public float Sum()
    {
        float sum = 0;
        for (int i = 0; i < Size; i++)
        {
            sum += data[i];
        }
        return sum;
    }

    public float MagnitudeSquared()
    {
        float sum = 0;
        for (int i = 0; i < Size; i++)
        {
            sum += data[i] * data[i];
        }
        return sum;
    }
}