####################### importing Libraries #######################
import streamlit as st
import torch
import classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################## Model Path  #######################
def find_model_path(model_type, e, b, h, h2 = 0):
    if model_type==0:
        return f"/Users/nimitt/Documents/ML/ML-ES335/q1/model_states/model_{b}_{e}_{h}.pt"
    elif model_type==1:
        return f"/Users/nimitt/Documents/ML/ML-ES335/q1/model_states/model_{b}_{e}_{h}_{h2}.pt"
    else :
        return "/Users/nimitt/Documents/ML/ML-ES335/q1/model_states/model2.pt"
    
########################## Streamlit App ##############################
def main():
    st.title('Text Generation')

    seed = st.slider("Seed",0,100,50,step = 1)
    g = torch.Generator()
    g.manual_seed(seed) 


    block_size_dict = {10:10,15:15,20:20,25:25}
    embedding_size_dict = {4:4,6:6,8:8,10:10,12:12}
    model_types = {"MLP; layers : 2":0, "MLP; layers : 3":1}

    stored_X_tensors = torch.load("/Users/nimitt/Documents/ML/ML-ES335/q1/model_states/10.pt")
    stoi,itos =  stored_X_tensors['stoi'], stored_X_tensors['itos'] 

    model_type = model_types[st.selectbox("Model Type",list(model_types.keys()))]
    block_size = st.selectbox("Context Length", list(block_size_dict.keys()))
    embedding_size = st.selectbox("Embedding Size", list(embedding_size_dict.keys()))
    generate_text_len = st.slider('Lenght of generated text', 0, 100, 50, step=1)
    

    st.write('Enter some text and click on Predict button')
    input_text = st.text_area("Enter text...")
    input_text = input_text.lower()
    input_text = input_text.replace("\n","~")

    hidden_size = 100
    hidden_size_2 = 50

    if model_type == 0:
        model = classes.NextChar(block_size,len(stoi),embedding_size,hidden_size)
    elif model_type == 1:
        model = classes.NextCharDense(block_size,len(stoi),embedding_size,hidden_size,hidden_size_2)
    else:
        model = classes.NextChar(block_size,len(stoi),embedding_size,hidden_size)

    model_path = find_model_path(model_type, embedding_size, block_size, hidden_size,hidden_size_2)
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    classes.load_check_points(model,opt,model_path)

    if st.button("Predict"):
        if input_text:
            # Make predictions
            # text = classes.generate_text(seed, model, input_text, itos, stoi, block_size, generate_text_len)

            g = torch.Generator()
            g.manual_seed(seed)
            context = []
            for j in range(len(input_text)):
                context = context + [stoi[input_text[j]]]
            if len(context) > block_size:
                context = context[-block_size:]
            if len(context) < block_size:
                while(len(context)!=block_size):
                    context.insert(0,0)
                
            text = ''
            for i in range(generate_text_len):
                x = torch.tensor(context).view(1, -1).to(device)
                y_pred = model(x)
                y_pred = y_pred.reshape(-1)
                k = seed//3
                top_k_values, top_k_indices = torch.topk(y_pred, k)
                ix = top_k_indices[k-1].item() 
                ch = itos[ix]
                # if ch == '~':
                #     break
                text += ch
                context = context[1:] + [ix]

            text = text.replace('~','\n')
            st.write(f'{text}')
        else:
            st.write("Please enter some text to predict.")

if __name__ == '__main__':
    main()