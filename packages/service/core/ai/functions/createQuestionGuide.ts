import type { ChatCompletionMessageParam } from '@fastgpt/global/core/ai/type.d';
import { getAIApi } from '../config';
import { countGptMessagesTokens } from '../../../common/string/tiktoken/index';

export const Prompt_QuestionGuide = `You are an AI intelligent assistant that can answer and solve my questions. Please generate 3 questions for me in conjunction with the previous dialog record to guide me to continue asking questions. The questions should be less than 20 characters long and returned in JSON format: [“Question 1”, “Question 2”, “Question 3”]`;

export async function createQuestionGuide({
  messages,
  model
}: {
  messages: ChatCompletionMessageParam[];
  model: string;
}) {
  const concatMessages: ChatCompletionMessageParam[] = [
    ...messages,
    {
      role: 'user',
      content: Prompt_QuestionGuide
    }
  ];
  const ai = getAIApi({
    timeout: 480000
  });
  const data = await ai.chat.completions.create({
    model: model,
    temperature: 0.1,
    max_tokens: 200,
    messages: concatMessages,
    stream: false
  });

  const answer = data.choices?.[0]?.message?.content || '';

  const start = answer.indexOf('[');
  const end = answer.lastIndexOf(']');

  const tokens = await countGptMessagesTokens(concatMessages);

  if (start === -1 || end === -1) {
    return {
      result: [],
      tokens: 0
    };
  }

  const jsonStr = answer
    .substring(start, end + 1)
    .replace(/(\\n|\\)/g, '')
    .replace(/  /g, '');

  try {
    return {
      result: JSON.parse(jsonStr),
      tokens
    };
  } catch (error) {
    return {
      result: [],
      tokens: 0
    };
  }
}
