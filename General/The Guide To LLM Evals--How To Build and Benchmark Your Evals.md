---
date: 2024-01-28
time: 09:22
note_maturity: 🌱
tags:
  - medium
  - article
link: https://towardsdatascience.com/llm-evals-setup-and-the-metrics-that-matter-2cc27e8e35f3
---
# The Guide To LLM Evals: How To Build and Benchmark Your Evals

_This piece is co-authored by_ [_Ilya Reznik_](https://ibreznik.com/)

Large language models (LLMs) are an incredible tool for developers and business leaders to create new value for consumers. They make personal recommendations, translate between unstructured and structured data, summarize large amounts of information, and do so much more.

As the applications multiply, so does the importance of measuring the performance of LLM-based applications. This is a nontrivial problem for several reasons: user feedback or any other **“source of truth” is extremely limited and often nonexistent**; even when possible, **human labeling is still expensive;** and it is easy to make these applications **complex**.

This complexity is often hidden by the abstraction layers of code and only becomes apparent when things go wrong. One line of code can initiate a cascade of calls (spans). **Different evaluations are required for each span**, thus multiplying your problems. For example, the simple code snippet below triggers multiple sub-LLM calls.

![](https://miro.medium.com/v2/resize:fit:700/1*Of9GvMKKw5JIvJxoEbyjZQ.jpeg)

Diagram by author

Fortunately, we can use the power of LLMs to automate the evaluation. In this article, we will delve into how to set this up and make sure it is reliable.

**The core of LLM evals is AI evaluating AI.**

While this may sound circular, we have always had human intelligence evaluate human intelligence (for example, at a job interview or your college finals). Now AI systems can finally do the same for other AI systems.

The process here is for LLMs to generate synthetic ground truth that can be used to evaluate another system. Which begs a question: why not use human feedback directly? Put simply, because you will never have enough of it.

Getting human feedback on even one percent of your input/output pairs is a gigantic feat. Most teams don’t even get that. But in order for this process to be truly useful, it is important to have evals on every LLM sub-call, of which we have already seen there can be many.

Let’s explore how to do this.

# LLM Model Evaluation vs. LLM System Evaluation

LLM_model_evals != LLM_System_evals

## LLM Model Evals

You might have heard of LLM evals. This term gets used in many different ways that all sound very similar but actually are very different. One of the more common ways it gets used is in what we will call **LLM model evals**. LLM model evals are focused on the overall performance of the foundational models. The companies launching the original customer-facing LLMs needed a way to quantify their effectiveness across an array of different tasks.

![](https://miro.medium.com/v2/resize:fit:700/1*y_Wxfs2gyZ3mVMoI2DPHTA.jpeg)

Diagram by author | In this case, we are evaluating two different open source foundation models. We are testing the same dataset across the twomodels and seeing how their metrics, like hellaswag or mmlu, stack up.

One popular library that has LLM model evals is the [OpenAI Eval library](https://towardsdatascience.com/how-to-best-leverage-openais-evals-framework-c38bcef0ec47), which was originally focused on the model evaluation use case. There are many metrics out there, like [HellaSwag](https://arxiv.org/abs/1905.07830) (which evaluates how well an LLM can complete a sentence), [TruthfulQA](https://arxiv.org/abs/2109.07958) (measuring truthfulness of model responses), and [MMLU](https://arxiv.org/abs/2009.03300) (which measures how well the LLM can multitask). There are even [LLM leaderboards](https://arize.com/blog-course/llm-leaderboards-benchmarks/) that looks at how well the open-source LLMs stack up against each other.

## LLM System Evals

Up to this point, we have discussed LLM model evaluation. In contrast, **LLM system evaluation** is the complete evaluation of components that you have control of in your system. The most important of these components are the prompt (or [prompt template](https://arize.com/blog/prompt-templates-functions-and-prompt-window-management/)) and context. ==LLM system evals assess how well your inputs can determine your outputs.==

LLM system evals may, for example, hold the LLM constant and change the prompt template. Since prompts are more dynamic parts of your system, this evaluation makes a lot of sense throughout the lifetime of the project. For example, an LLM can evaluate your chatbot responses for usefulness or politeness, and the same eval can give you information about performance changes over time in production.

![](https://miro.medium.com/v2/resize:fit:700/1*iVwTDexiQruQoMOp2ncl3Q.jpeg)

Diagram by author | _In this case, we are evaluating two different prompt templates on a single foundation model. We are testing the same dataset across the two templates and seeing how their metrics like precision and recall stack up._

## Which To Use? It Depends On Your Role

There are distinct personas who make use of LLM evals. One is the model developer or an engineer tasked with fine-tuning the core LLM, and the other is the practitioner assembling the user-facing system.

There are very few LLM model developers, and they tend to work for places like OpenAI, Anthropic, Google, Meta, and elsewhere. **Model developers care about LLM model evals,** as their job is to deliver a model that caters to a wide variety of use cases.

For ML practitioners, the task also starts with model evaluation. One of the first steps in developing an LLM system is picking a model (i.e. GPT 3.5 vs 4 vs Palm, etc.). The LLM model eval for this group, however, is often a one-time step. Once the question of which model performs best in your use case is settled, the majority of the rest of the application’s lifecycle will be defined by LLM system evals. Thus, **ML practitioners care about both LLM model evals and LLM system evals but likely spend much more time on the latter**.

# LLM System Eval Metrics Vary By Use Case

Having worked with other ML systems, your first question is likely this: “What should the outcome metric be?” The answer depends on what you are trying to evaluate.

- **Extracting structured information**: You can look at how well the LLM extracts information. For example, you can look at completeness (is there information in the input that is not in the output?).
- **Question answering**: How well does the system answer the user’s question? You can look at the accuracy, politeness, or brevity of the answer — or all of the above.
- **Retrieval Augmented Generation (RAG)**: Are the retrieved documents and final answer relevant?

As a system designer, you are ultimately responsible for system performance, and so it is up to you to understand which aspects of the system need to be evaluated. For example, If you have an LLM interacting with children, like a tutoring app, you would want to make sure that the responses are age-appropriate and are not toxic.

Some common evaluations being employed today are relevance, hallucinations, question-answering accuracy, and toxicity. Each one of these evals will have different templates based on what you are trying to evaluate. Here is an example with relevance:

This example uses the open-source [Phoenix tool](https://github.com/Arize-ai/phoenix) for simplicity (full disclosure: I am on the team that developed Phoenix). Within the Phoenix tool, there exist default templates for most common use cases. Here is the one we will use for this example:

You are comparing a reference text to a question and trying to determine if the reference text contains information relevant to answering the question. Here is the data:  
    [BEGIN DATA]  
    ************  
    [Question]: {query}  
    ************  
    [Reference text]: {reference}  
    [END DATA]  
Compare the Question above to the Reference text. You must determine whether the Reference text  
contains information that can answer the Question. Please focus on whether the very specific  
question can be answered by the information in the Reference text.  
Your response must be single word, either "relevant" or "irrelevant",  
and should not contain any text or characters aside from that word.  
"irrelevant" means that the reference text does not contain an answer to the Question.  
"relevant" means the reference text contains an answer to the Question.

We will also use OpenAI’s GPT-4 model and scikitlearn’s precision/recall metrics.

First, we will import all necessary dependencies:

from phoenix.experimental.evals import (  
   RAG_RELEVANCY_PROMPT_TEMPLATE_STR,  
   RAG_RELEVANCY_PROMPT_RAILS_MAP,  
   OpenAIModel,  
   download_benchmark_dataset,  
   llm_eval_binary,  
)  
from sklearn.metrics import precision_recall_fscore_support

Now, let’s bring in the dataset:

# Download a "golden dataset" built into Phoenix  
benchmark_dataset = download_benchmark_dataset(  
   task="binary-relevance-classification", dataset_name="wiki_qa-train"  
)  
# For the sake of speed, we'll just sample 100 examples in a repeatable way  
benchmark_dataset = benchmark_dataset.sample(100, random_state=2023)  
benchmark_dataset = benchmark_dataset.rename(  
   columns={  
       "query_text": "query",  
       "document_text": "reference",  
   },  
)  
# Match the label between our dataset and what the eval will generate  
y_true = benchmark_dataset["relevant"].map({True: "relevant", False: "irrelevant"})

Now let’s conduct our evaluation:

# Any general purpose LLM should work here, but it is best practice to keep the temperature at 0  
model = OpenAIModel(  
   model_name="gpt-4",  
   temperature=0.0,  
)  
# Rails will define our output classes  
rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())  
  
  
benchmark_dataset["eval_relevance"] = \  
   llm_eval_binary(benchmark_dataset,  
                   model,  
                   RAG_RELEVANCY_PROMPT_TEMPLATE_STR,  
                   rails)  
y_pred = benchmark_dataset["eval_relevance"]  
  
  
# Calculate evaluation metrics  
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

## Evaluating LLM-Based Systems with LLMs

There are two distinct steps to the process of evaluating your LLM-based system with an LLM. First, establish a benchmark for your LLM evaluation metric. To do this, you put together a dedicated LLM-based eval whose only task is to label data as effectively as a human labeled your “golden dataset.” You then benchmark your metric against that eval. Then, run this LLM evaluation metric against results of your LLM application (more on this below).

# How To Build An LLM Eval

The first step, as we covered above, is to build a benchmark for your evaluations.

To do that, you must begin with a **metric best suited for your use case**. Then, you need the **golden dataset**. This should be representative of the type of data you expect the LLM eval to see. The golden dataset should have the “ground truth” label so that we can measure performance of the LLM eval template. Often such labels come from human feedback. Building such a dataset is laborious, but you can often find a standardized one for the most common use cases (as we did in the code above).

![](https://miro.medium.com/v2/resize:fit:700/1*IQ6Yc_7XnuBwsQ84jYKJGw.jpeg)

Diagram by author

Then you need to decide **which LLM** you want to use for evaluation. This could be a different LLM from the one you are using for your application. For example, you may be using Llama for your application and GPT-4 for your eval. Often this choice is influenced by questions of cost and accuracy.

![](https://miro.medium.com/v2/resize:fit:700/1*VVkotludDAjiOOC8PgEmag.jpeg)

Diagram by author

Now comes the core component that we are trying to benchmark and improve: the **eval template**. If you’re using an existing library like OpenAI or Phoenix, you should start with an existing template and see how that prompt performs.

If there’s a specific nuance you want to incorporate, adjust the template accordingly or build your own from scratch.

Keep in mind that the template should have a clear structure, like the one we used in prior section. Be explicit about the following:

- **What is the input?** In our example, it is the documents/context that was retrieved and the query from the user.
- **What are we asking?** In our example, we’re asking the LLM to tell us if the document was relevant to the query
- **What are the possible output formats?** In our example, it is binary relevant/irrelevant, but it can also be multi-class (e.g., fully relevant, partially relevant, not relevant).

![](https://miro.medium.com/v2/resize:fit:700/1*3Uf95EWIRNP9vQUDNQmpQw.jpeg)

Diagram by author

You now need to run the eval across your golden dataset. Then you can **generate metrics** (overall accuracy, precision, recall, F1, etc.) to determine the benchmark. It is important to look at more than just overall accuracy. We’ll discuss that below in more detail.

If you are not satisfied with the performance of your LLM evaluation template, you need to change the prompt to make it perform better. This is an iterative process informed by hard metrics. As is always the case, it is important to avoid overfitting the template to the golden dataset. Make sure to have a representative holdout set or run a k-fold cross-validation.

![](https://miro.medium.com/v2/resize:fit:700/1*ikAemfhUyw04SnnlCFHnbg.jpeg)

Diagram by author

Finally, you arrive at your **benchmark**. The optimized performance on the golden dataset represents how confident you can be on your LLM eval. It will not be as accurate as your ground truth, but it will be accurate enough, and it will cost much less than having a human labeler in the loop on every example.

Preparing and customizing your prompt templates allows you to set up test cases.

# Why You Should Use Precision and Recall When Benchmarking Your LLM Prompt Template

The industry has not fully standardized best practices on LLM evals. Teams commonly do not know how to establish the right benchmark metrics.

Overall accuracy is used often, but it is not enough.

This is one of the most common problems in data science in action: very significant class imbalance makes accuracy an impractical metric.

Thinking about it in terms of the relevance metric is helpful. Say you go through all the trouble and expense of putting together the most relevant chatbot you can. You pick an LLM and a template that are right for the use case. This should mean that significantly more of your examples should be evaluated as “relevant.” Let’s pick an extreme number to illustrate the point: 99.99% of all queries return relevant results. Hooray!

Now look at it from the point of view of the LLM eval template. If the output was “relevant” in all cases, without even looking at the data, it would be right 99.99% of the time. But it would simultaneously miss all of the (arguably most) important cases — ones where the model returns irrelevant results, which are the very ones we must catch.

In this example, accuracy would be high, but [precision and recall](https://arize.com/blog-course/precision-vs-recall/) (or a combination of the two, like the [F1 score](https://arize.com/blog-course/f1-score/)) would be very low. Precision and recall are a better measure of your model’s performance here.

The other useful visualization is the confusion matrix, which basically lets you see correctly and incorrectly predicted percentages of relevant and irrelevant examples.

![](https://miro.medium.com/v2/resize:fit:700/1*EAjzWgBkAljSDNAKbl5epA.png)

Diagram by author | _In this example, we see that the highest percentage of predictions are correct: a relevant example in the golden dataset has an 88% chance of being labeled as such by our eval. However, we see that the eval performs significantly worse on “irrelevant” examples, mislabeling them more than 27% of the time._

# How To Run LLM Evals On Your Application

At this point you should have both your model and your tested LLM eval. You have proven to yourself that the eval works and have a quantifiable understanding of its performance against the ground truth. Time to build more trust!

Now we can actually use our eval on our application. This will help us measure how well our LLM application is doing and figure out how to improve it.

![](https://miro.medium.com/v2/resize:fit:700/1*dA6CjLdWbZRjBG6HWEMCuQ.jpeg)

Diagram by author

The LLM system eval runs your entire system with one extra step. For example:

- You retrieve your input docs and add them to your prompt template, together with sample user input.
- You provide that prompt to the LLM and receive the answer.
- You provide the prompt and the answer to your eval, asking it if the answer is relevant to the prompt.

It is a best practice not to do LLM evals with one-off code but rather a library that has built-in prompt templates. This increases reproducibility and allows for more flexible evaluation where you can swap out different pieces.

These evals need to work in three different environments:

## Pre-production

When you’re doing the benchmarking.

## Pre-production

When you’re testing your application. This is somewhat similar to the offline evaluation concept in traditional ML. The idea is to understand the performance of your system before you ship it to customers.

## Production

When it’s deployed. Life is messy. Data drifts, users drift, models drift, all in unpredictable ways. Just because your system worked well once doesn’t mean it will do so on Tuesday at 7 p.m. Evals help you continuously understand your system’s performance after deployment.

![](https://miro.medium.com/v2/resize:fit:700/1*n1X9l1i1l-YtsVbhS43ruA.jpeg)

Diagram by author

# Questions To Consider

## **How many rows should you sample?**

The LLM-evaluating-LLM paradigm is not magic. You cannot evaluate every example you have ever run across — that would be prohibitively expensive. However, you already have to sample data during human labeling, and having more automation only makes this easier and cheaper. So you can sample more rows than you would with human labeling.

## **What evals should you use?**

This depends largely on your use case. For search and retrieval, relevancy-type evals work best. Toxicity and hallucinations have specific eval patterns (more on that above).

Some of these evals are important in the troubleshooting flow. Question-answering accuracy might be a good overall metric, but if you dig into why this metric is underperforming in your system, you may discover it is because of bad retrieval, for example. There are often many possible reasons, and you might need multiple metrics to get to the bottom of it.

## **What model should you use?**

It is impossible to say that one model works best for all cases. Instead, you should run model evaluations to understand which model is right for your application. You may also need to consider tradeoffs of recall vs. precision, depending on what makes sense for your application. In other words, do some data science to understand this for your particular case.

![](https://miro.medium.com/v2/resize:fit:700/1*pkiSUxHhIjU5lwzZdHDHrA.jpeg)

Diagram by author

# Conclusion

Being able to evaluate the performance of your application is very important when it comes to production code. In the era of LLMs, the problems have gotten harder, but luckily we can use the very technology of LLMs to help us in running evaluations. [LLM evaluation](https://arize.com/blog-course/llm-evaluation-the-definitive-guide/) should test the whole system and not just the underlying LLM model — think about how much a prompt template matters to user experience. Best practices, standardized tooling, and curated datasets simplify the job of developing LLM systems.










#### 🧭  Idea Compass
- West  (similar) 
[[LLM]] [[Model Evaluation]]
- East (opposite)
	- 
- North (theme/question)
	- how to evaluate LLM Systems
- South (what follows)
	- 