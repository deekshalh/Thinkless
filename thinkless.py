!pip install pandas transformers[torch] accelerate
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Define your custom dataset class
class CustomSummarizationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.iloc[idx, 0]  # Assuming the source text is in the first column
        target_text = self.data.iloc[idx, 1]  # Assuming the target summary is in the second column

        source_encoding = self.tokenizer(
            source_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_prefix_space=True
        )

        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# Define your tokenizer and dataset
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
dataset = CustomSummarizationDataset('/content/Copy of Dataset personal - Sheet1.csv', tokenizer, max_length=512)

# Load the pre-trained model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Generate a summary
input_text = "1.1 A SIMPLE ECONOMY Think of any society. People in the society need many goods and services in their everyday life including food, clothing, shelter, transport facilities like roads and railways, postal services and various other services like that of teachers and doctors. In fact, the list of goods and services that any individual needs is so large that no individual in society, to begin with, has all the things she needs. Every individual has some amount of only a few of the goods and services that she would like to use. A family farm may own a plot of land, some grains, farming implements, maybe a pair of bullocks and also the labour services of the family members. A weaver may have some yarn, some cotton and other instruments required for weaving cloth. The teacher in the local school has the skills required to impart education to the students. Some others in society may not have any resource3 accepting their own labour services. Each of these decision making units can produce some goods or services by using the resources that it has and use part of the produce to obtain the many other goods and services which it needs. For example, the family farm can produce corn, use part of the produce for consumption purposes and procure clothing, housing and various services in exchange for the rest of the produce. Similarly, the weaver can get the goods and services that she wants in exchange for the cloth she produces in her yarn. The teacher can earn some money by teaching students in the school and use the money for obtaining the goods and services that she wants. The labourer also can try to fulfil her needs by using whatever money she can earn by working for someone else. Each individual can thus use her resources to fulfil her needs. It goes without saying that no individual has unlimited resources compared to her needs. The amount of corn that the family farm can produce is limited by the amount of resources it has, and hence, the amount of different goods and services that it can procure in exchange for corn is also limited. As a result, the family is forced to make a choice between the different goods and services that are available. It can have more of a good or service only by giving up some amounts of other goods or services. For example, if the family wants to have a bigger house, it may have to give up the idea of having a few more acres of arable land. If it wants more and better education for the children, it may have to give up some of the luxuries of life. The same is the case with all other individuals in society. Everyone faces scarcity of resources, and therefore, has to use the limited resources in the best possible way to fulfil her needs. In general, every individual in society is engaged in the production of some goods or services and she wants a combination of many goods and services not all of which are produced by her. Needless to say that there has to be some compatibility between what people in society collectively want to have and what they produce4 . For example, the total amount of corn produced by a family farm along with other farming units in a society must match the total amount of corn that people in the society collectively want to consume. If people in the society do not want as much corn as the farming units are capable of producing collectively, a part of the resources of these units could have been used in the production of some other goods or services which are in high demand. On the other hand, if people in the society want more corn compared to what the farming units are producing collectively, the resources used in the production of some other goods and services may be reallocated to the production of corn. Similar is the case with all other goods or services. Just as the resources of an individual are scarce, the resources of the society are also scarce in comparison to what the people in the society might collectively want to have. The scarce resources of the society have to be allocated properly in the production of different goods and services in keeping with the likes and dislikes of the people of the society. Any allocation of resources of the society would result in the production of a particular combination of different goods and services. The goods and services thus produced will have to be distributed among the individuals of the society. The allocation of the limited resources and the distribution of the final mix of goods and services are two of the basic economic problems faced by the society. In reality, any economy is much more complex compared to the society discussed above. In the light of what we have learnt about the society, let us now discuss the fundamental concerns of the discipline of economics some of which we shall study throughout this book"  # Replace with your input text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
