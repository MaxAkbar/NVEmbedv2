using OpenAI;
using OpenAI.Embeddings;
using System.ClientModel;

// Initialize the OpenAI API client
var openAIOptions = new OpenAIClientOptions
{
    Endpoint = new Uri("http://127.0.0.1:5000/v1")
};
var apiKey = new ApiKeyCredential("FAKE_API");
var model = "NV-Embed-v2";
var embeddingClient = new EmbeddingClient(model, apiKey, openAIOptions);
// Define the input text for which you want to generate embeddings
var inputText = "Hello, World!";

OpenAIEmbedding embedding = embeddingClient.GenerateEmbedding(inputText);
ReadOnlyMemory<float> vector = embedding.ToFloats();

// Output the embedding vector
Console.WriteLine($"Embedding for {inputText}:");
Console.WriteLine($"Dimension: {vector.Length}");
Console.WriteLine($"Floats: ");
// get the first 10 elements of the embedding vector
for (int i = 0; i < 10; i++)
{
    Console.WriteLine($"  [{i,4}] = {vector.Span[i]}");
}