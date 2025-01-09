from groq import Groq
import gradio as gr
import os
os.environ["GROQ_API_KEY"]="your_api_key"

def is_query_relevant_to_image(image_url, query):
    if 'image' not in query.lower():
        return False
    return True

def generate_response(input_data):
    client = Groq()  
    messages = []

    if 'image_url' in input_data and 'query' not in input_data:
        messages.append(
            {
                "role": "user",
                "content": "Generate Image caption in less than 50 words "
            }
        )
        messages.append(
            {
                "role": "user",
                "content": input_data['image_url']
            }
        )
        
    elif 'image_url' in input_data and 'query' in input_data:
        if not is_query_relevant_to_image(input_data['image_url'], input_data['query']):
            return "Error: The query doesn't seem to match the image. Please provide a relevant query."
        messages.append(
            {
                "role": "user",
                "content": input_data['image_url']
            }
        )
        messages.append(
            {
                "role": "user",
                "content": input_data['query']
            }
        )

    elif 'query' in input_data and 'image_url' not in input_data:
        messages.append(
            {
                "role": "user",
                "content": input_data['query']
            }
        )

    if not messages:
        raise ValueError("Input data does not contain valid content.")

    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=messages,
        temperature=0.7,        
        max_tokens=300,         
        top_p=0.9,              
        stream=False,           
        stop=None               
    )

    return completion.choices[0].message


def process_query_and_image(image_url=None, query=None):

    input_data = {}

    if image_url:
        input_data['image_url'] = image_url

    if query:
        input_data['query'] = query

    if not input_data:
        return "Please provide either an image URL, a query, or both."

    try:
        if 'image_url' in input_data and 'query' in input_data:
            return generate_response({"image_url": input_data['image_url'], "query": input_data['query']})
        return generate_response(input_data)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Welcome to the Image Query and Caption Generator!")
    print("You can provide an image URL, a query, or both to generate a response.\n")

    image_url = input("Enter the image URL (or leave blank to skip): ").strip()
    query = input("Enter your query (or leave blank to skip): ").strip()

    if not image_url and not query:
        print("\nError: You must provide at least an image URL or a query.")
    else:
        print("\nResponse:")
        print(process_query_and_image(image_url if image_url else None, query if query else None))

demo = gr.Interface(
    fn=process_query_and_image,
    inputs=[
        gr.Textbox(label="Image URL", placeholder="Enter image URL (optional)"),
        gr.Textbox(label="Query", placeholder="Enter your query (optional)")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Image Query and Caption Generator",
    description="Provide an image URL and/or a query to get a response. If the query doesn't match the image, you will be informed."
)
demo.launch()

