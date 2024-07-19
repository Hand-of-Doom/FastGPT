import { replaceVariable } from '@fastgpt/global/common/string/tools';
import { getAIApi } from '../config';
import { ChatItemType } from '@fastgpt/global/core/chat/type';
import { countGptMessagesTokens } from '../../../common/string/tiktoken/index';
import { ChatCompletionMessageParam } from '@fastgpt/global/core/ai/type';
import { chatValue2RuntimePrompt } from '@fastgpt/global/core/chat/adapt';

/* 
    query extension - 问题扩展
    可以根据上下文，消除指代性问题以及扩展问题，利于检索。
*/

const defaultPrompt = `As a vector retrieval assistant, your task is to generate a different version of the “search term” for the “original question” from different perspectives, so as to improve the semantic richness of the vector retrieval and the accuracy of the vector retrieval. The generated question should point to a clear object and be in the same language as the original question. For example:
History. 
“"”
““””
Original question: Introduce the plot.
Search terms: ["Introduce the background of the story.” , “What is the theme of the story?” , “Introduce the main characters of the story.”]
----------------
History. 
“"”
Q: Context of the conversation.
A: The current dialog is about Nginx introduction, usage etc.
“"”
Original question: How to download
Search terms: ["How to download Nginx?” , “What do I need to download Nginx?” , “What are the channels to download Nginx?”]
----------------
History. 
“"”
Q: Context of the conversation.
A: The current conversation is about Nginx and how to use it.
Q: Error “no connection”
A: The error “no connection” may be caused by ......
“"”
Original question: How to solve
Search term: [“How to fix Nginx error ‘no connection’?”], “How to fix Nginx error ‘no connection’? , “Causes of ‘no connection’ error.” , “Nginx is reporting ‘no connection’, what should I do?”]
----------------
History. 
“"”
Q: How many days is the maternity leave?
A: The number of days of maternity leave depends on the city of the employee. Please provide your city so I can answer your question.
“"”
Original Question: Shenyang
Search Terms: ["How many days of maternity leave is granted in Shenyang?” , “Shenyang's Nursing Maternity Leave Policy.” , “Shenyang's Nursing Maternity Leave Standards.”"]
----------------
History. 
“"”
Q: Who is the author?
A: The author of FastGPT is labring.
“"”
Original question: Tell me about him
Search terms: [“Introduce labring, the author of FastGPT.”,“ Background information on author labring.” “,” Why does labring do FastGPT?"]
----------------
History.
“"”
Q: Conversation background.
A: Questions about the introduction and use of FatGPT.
“"”
Original question: Hello.
Search term: [“Hello”]
----------------
History.
“"”
Q: How does FastGPT charge?
A: FastGPT charges can be found at ......
“"”
Original question: Do you know laf?
Search term: [“What is laf's official website address?”] , “laf Tutorials. , “Tutorial on how to use laf.” , “What are the features and advantages of laf.”]
----------------
History.
“"”
Q: Advantages of FastGPT
A: 1. open source
   2. Simplicity
   3. Extensible
“"”
Original question: Introduce point 2.
Search terms: [“Introduce the advantages of FastGPT's simplicity”, “In what ways can FastGPT's simplicity be demonstrated”].
----------------
History.
“"”
Q: What is FastGPT?
A: FastGPT is a RAG platform.
Q: What is Laf?
A: Laf is a cloud function development platform.
“"”
Original Question: How are they related?
Search Terms: [“What is the relationship between FastGPT and Laf?”], “Introducing FastGPT. , “Introducing FastGPT”, “Introducing Laf”]
----------------
History.
“"”
{{histories}}
“"”
Original question: {{query}}
Search terms: {{histories}} `;

export const queryExtension = async ({
  chatBg,
  query,
  histories = [],
  model
}: {
  chatBg?: string;
  query: string;
  histories: ChatItemType[];
  model: string;
}): Promise<{
  rawQuery: string;
  extensionQueries: string[];
  model: string;
  tokens: number;
}> => {
  const systemFewShot = chatBg
    ? `Q: Conversation context.
A: ${chatBg}
`
    : '';
  const historyFewShot = histories
    .map((item) => {
      const role = item.obj === 'Human' ? 'Q' : 'A';
      return `${role}: ${chatValue2RuntimePrompt(item.value).text}`;
    })
    .join('\n');
  const concatFewShot = `${systemFewShot}${historyFewShot}`.trim();

  const ai = getAIApi({
    timeout: 480000
  });

  const messages = [
    {
      role: 'user',
      content: replaceVariable(defaultPrompt, {
        query: `${query}`,
        histories: concatFewShot
      })
    }
  ] as ChatCompletionMessageParam[];
  const result = await ai.chat.completions.create({
    model: model,
    temperature: 0.01,
    // @ts-ignore
    messages,
    stream: false
  });

  let answer = result.choices?.[0]?.message?.content || '';
  if (!answer) {
    return {
      rawQuery: query,
      extensionQueries: [],
      model,
      tokens: 0
    };
  }

  answer = answer.replace(/\\"/g, '"');

  try {
    const queries = JSON.parse(answer) as string[];

    return {
      rawQuery: query,
      extensionQueries: Array.isArray(queries) ? queries : [],
      model,
      tokens: await countGptMessagesTokens(messages)
    };
  } catch (error) {
    console.log(error);
    return {
      rawQuery: query,
      extensionQueries: [],
      model,
      tokens: 0
    };
  }
};
