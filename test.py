  import argparse
  import base64
  import os
  import re
  from io import BytesIO

  # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  import openai
  import requests
  from PIL import Image
  from tqdm import tqdm

  from visual_scoring.score import (
      UnifiedQAModel,
      VQAModel,
      VS_score_single,
      filter_question_and_answers,
      get_question_and_answers,
  )

  # 传入参数

  data_type = "shape"
  device = f"cuda:{0}"
  begin_idx = 0
  print(f"Data type: {data_type}")
  print(f"Device: {device}")
  print(f"Begin index: {begin_idx}")

  api_key = "sk-M5ppriS3vTYSiwFn3c58Af766d7c4956B4EcEc36888a1c2b"
  api_base = "https://ai98.vip/v1"
  os.environ["OPENAI_API_KEY"] = api_key
  os.environ["OPENAI_API_BASE"] = api_base
  openai.api_key = api_key
  openai.base_url = api_base
  client = openai.OpenAI(api_key=api_key, base_url=api_base)

  def openai_completion(prompt, engine="gpt-4o", max_tokens=700, temperature=0):
      resp = client.chat.completions.create(
          model=engine,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=max_tokens,
          temperature=temperature,
          stop=["\n\n", "<|endoftext|>"],
      )

      return resp.choices[0].message.content

  def get_image_from_url(url: str):
      response = requests.get(url)
      img = Image.open(BytesIO(response.content))
      img = img.resize((224, 224))
      img = img.convert("RGB")
      return img


  def get_image_from_path(file_path: str):
      img = Image.open(file_path)
      img = img.resize((224, 224))
      img = img.convert("RGB")
      return img


  def encode_image_from_path(image_path):
      """
      对图片文件进行 Base64 编码

      输入：
          - image_path：图片的文件路径
      输出：
          - 编码后的 Base64 字符串
      """
      # 二进制读取模式打开图片文件，
      with open(image_path, "rb") as image_file:
          # 将编码后的字节串解码为 UTF-8 字符串，以便于在文本环境中使用。
          return base64.b64encode(image_file.read()).decode("utf-8")


  def encode_image_from_PIL_image(image):
      # 创建一个内存缓冲区
      buffered = BytesIO()
      # 将 PIL 图像对象保存到内存缓冲区中，格式为 JPEG，你也可以选择其他格式
      image.save(buffered, format="JPEG")
      # 获取缓冲区中的字节数据并将其编码为 base64 字符串
      img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
      return img_str


  unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200", device=device)
  vqa_model = VQAModel("mplug-large", device=device)

  gpt_questions = get_question_and_answers("One full pitcher of beer with an elephant's trunk in it.")
  gpt_questions

  def get_VS_result(text, img_path, filtered_questions=None):
      if not filtered_questions:
          # Generate questions with GPT
          gpt_questions = get_question_and_answers(text)

          # Filter questions with UnifiedQA
          filtered_questions = filter_question_and_answers(unifiedqa_model, gpt_questions)

          # See the questions
          # print(filtered_questions)

          # calucluate VS score
          result = VS_score_single(vqa_model, filtered_questions, img_path)
          return filtered_questions, result
      else:
          # calucluate VS score
          result = VS_score_single(vqa_model, filtered_questions, img_path)
          return result
      
  def generate_image(prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
      response = client.images.generate(
          model=model,
          prompt=prompt,
          size=size,
          quality=quality,
          n=n,
      )

      image_url = response.data[0].url
      img = get_image_from_url(image_url)
      return img

  def format_prompt_to_message(
      user_prompt, previous_prompts, generated_image, num_solutions, result
  ):
      image = encode_image_from_PIL_image(generated_image)

      VS_results = []
      for i, (key, value) in enumerate(result["question_details"].items()):
          VS_result = "Element " + str(i) + "\n"
          VS_result += "Question: " + key + "\n"
          VS_result += "Ground Truth: " + value["answer"] + "\n"
          VS_result += (
              "In the image generated from above prompt, the VQA model identified infer that the answer to the question is: "
              + value["free_form_vqa"]
              + "\n"
          )

          VS_results.append(VS_result)

      VS_results = "\n".join(VS_results)

      prompt = f"""
  You are an expert prompt optimizer for text-to-image models. Text-to-image models take a text prompt as input and generate images depicting the prompt as output. You are responsible for transforming human-written prompts into improved prompts for text-to-image models. Your responses should be concise and effective.

  Your task is to optimize the human initial prompt: "{user_prompt}". Below are some previous prompts along with a breakdown of their visual elements. Each element is paired with a score indicating its presence in the generated image. A score of 1 indicates visual elements matching the human initial prompt, while a score of 0 indicates no match.

  Here is the image that the text-to-image model generated based on the initial prompt:
  {{image_placeholder}}

  Here are the previous prompts and their visual element scores:
  ## Previous Prompts
  {previous_prompts}
  ## Visual Element Scores
  {VS_results}

  Generate {num_solutions} paraphrases of the initial prompt which retain the semantic meaning and have higher scores than all the previous prompts. Prioritize optimizing for objects with the lowest scores. Prefer substitutions and reorderings over additions. Please respond with each new prompt in between <PROMPT> and </PROMPT>, for example:
  1. <PROMPT>paraphrase 1</PROMPT>
  2. <PROMPT>paraphrase 2</PROMPT>
  ...
  {num_solutions}. <PROMPT>paraphrase {num_solutions}</PROMPT>
  """
      text_prompts = prompt.split("{image_placeholder}")

      user_content = [{"type": "text", "text": text_prompts[0]}]
      base64_images = [
          {
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{image}",
                  "detail": "high",
              },
          }
      ]
      user_content.extend(base64_images)
      user_content.append({"type": "text", "text": text_prompts[1]})
      messages_template = [{"role": "user", "content": user_content}]

      return messages_template

  def generate_image_chat_response(messages_template, client):
      payload = {
          "model": "gpt-4o",
          "messages": messages_template,
          "max_tokens": 1600,
          "temperature": 0,
          "seed": 2024,
      }

      # 调用 OpenAI API 生成回复
      response = client.chat.completions.create(**payload)

      # 返回生成的结果
      return response.choices[0].message.content


  def extract_prompts(text):
      pattern = r"<PROMPT>(.*?)</PROMPT>"
      prompts = re.findall(pattern, text)
      return prompts

  max_retries = 10  # 最大重试次数


  def DALLE3_VS(prompt):
      success = False
      retries = 0
      print(f"Generating image for prompt: {prompt}")
      while not success and retries < max_retries:
          try:
              image = generate_image(prompt=prompt)
              success = True
              print("Image generated successfully!")
          except Exception as e:
              retries += 1
              print(f"Error: {e}")
              if retries < max_retries:
                  print(f"Retrying... ({retries}/{max_retries})")
                  # time.sleep(1)  # 等待 1 秒后重试
              else:
                  print("Max retries reached. Exiting.")
                  break
      if not success:
          print("Failed to generate image. Exiting.")
          return

      success = False
      retries = 0
      print("Calculating VS score...")
      while not success and retries < max_retries:
          try:
              filtered_questions, VS_result = get_VS_result(prompt, image)
              success = True
              print(f"\nVS score: {VS_result['VS_score']}")
          except Exception as e:
              retries += 1
              print(f"Error: {e}")
              if retries < max_retries:
                  print(f"Retrying... ({retries}/{max_retries})")
                  # time.sleep(1)  # 等待 1 秒后重试
              else:
                  print("Max retries reached. Exiting.")
                  break
      if not success:
          print("Failed to calculate VS score. Exiting.")
          return image

      success = False
      retries = 0
      print("Generating new prompt...")
      while not success and retries < max_retries:
          try:
              formatted_prompt = format_prompt_to_message(
                  user_prompt=prompt,
                  previous_prompts=prompt,
                  generated_image=image,
                  num_solutions=3,
                  result=VS_result,
              )
              generate_prompts = generate_image_chat_response(formatted_prompt, client)
              new_regional_prompt = extract_prompts(generate_prompts)[0]
              success = True
              print("Prompt formatted successfully!")
          except Exception as e:
              retries += 1
              print(f"Error: {e}")
              if retries < max_retries:
                  print(f"Retrying... ({retries}/{max_retries})")
                  # time.sleep(1)  # 等待 1 秒后重试
              else:
                  print("Max retries reached. Exiting.")
                  break
      if not success:
          print("Failed to generate new prompt. Exiting.")
          return image

      print(f"New prompt generated: {new_regional_prompt}")
      try:
          new_image = generate_image(
              prompt=prompt,
          )
      except Exception as e:
          print(f"Error: {e}")
          return image

      new_VS_result = get_VS_result(prompt, new_image, filtered_questions)
      print(f"\nVS score: {new_VS_result['VS_score']}")

      if new_VS_result["VS_score"] > VS_result["VS_score"]:
          return new_image
      else:
          return image
      
  prompt = "One full pitcher of beer with an elephant's trunk in it."
  DALLE3_VS(prompt)

  max_retries = 10  # 最大重试次数


  def generate_image_robust(prompt):
      success = False
      retries = 0
      print(f"Generating image for prompt: {prompt}")
      while not success and retries < max_retries:
          try:
              image = generate_image(prompt=prompt)
              success = True
              print("Image generated successfully!")
          except Exception as e:
              retries += 1
              print(f"Error: {e}")
              if retries < max_retries:
                  print(f"Retrying... ({retries}/{max_retries})")
                  # time.sleep(1)  # 等待 1 秒后重试
              else:
                  print("Max retries reached. Exiting.")
                  break
      if not success:
          print("Failed to generate image. Exiting.")
          raise Exception("Failed to generate image")
      else:
          print("Image generated successfully!")
          return image
      
  class Reviewer:
      """
      Agent A: 审阅者 (Reviewer)
      - 负责阅读/审阅初始解或思路，对其正确性、完整性进行评价；
      - 主要会找出优点与缺陷，但不一定提出深度质疑或修正方案。
      """
      
      def __init__(self):
          super().__init__()
          print("\nReviewer initialized.")
          print("----------------------")

      def calculate_VS_score(self, prompt, image):
          print("Calculating VS score...")
          success = False
          retries = 0
          print("Calculating VS score...")
          while not success and retries < max_retries:
              try:
                  filtered_questions, VS_result = get_VS_result(prompt, image)
                  success = True
                  print(f"\nVS score: {VS_result['VS_score']}")
              except Exception as e:
                  retries += 1
                  print(f"Error: {e}")
                  if retries < max_retries:
                      print(f"Retrying... ({retries}/{max_retries})")
                      # time.sleep(1)  # 等待 1 秒后重试
                  else:
                      print("Max retries reached. Exiting.")
                      break
          if not success:
              print("Failed to calculate VS score. Exiting.")
              raise Exception("Failed to calculate VS score.")
          else:
              print("VS score calculated successfully!")
              return filtered_questions, VS_result
          
          
      def format_prompt_to_message(
          self, user_prompt, previous_prompts, generated_image, vs_result
      ):
          image = encode_image_from_PIL_image(generated_image)

          VS_results = []
          for i, (key, value) in enumerate(vs_result["question_details"].items()):
              VS_result = "Element " + str(i) + "\n"
              VS_result += "Question: " + key + "\n"
              VS_result += "Ground Truth: " + value["answer"] + "\n"
              VS_result += (
                  "In the image generated from above prompt, the VQA model identified infer that the answer to the question is: "
                  + value["free_form_vqa"]
                  + "\n"
              )

              VS_results.append(VS_result)

          VS_results = "\n".join(VS_results)

          prompt = f"""
  You are a prompt reviewer for text-to-image models. Your role is to evaluate both the initial human-written prompt and previous prompts based on their effectiveness in conveying visual elements that match the generated images. Consider the scores assigned to each visual element in the outputs, with 1 indicating a perfect match and 0 indicating no match.

  Your task is to review the initial prompt: "{user_prompt}". Additionally, provide an evaluation of the previous prompts given.

  Here is the image that the text-to-image model generated based on the initial prompt:
  {{image_placeholder}}

  Here are the previous prompts and their visual element scores:
  ## Previous Prompts
  {previous_prompts}
  ## Visual Element Scores
  {VS_results}

  Provide a comprehensive evaluation of the initial prompt and each of the previous prompts. Focus on the correctness and completeness of each prompt in relation to the generated images, highlighting strengths and weaknesses. Depth questioning or suggested alterations are not necessary, but insightful commentary is encouraged.
  If there are no previous prompts, simply provide an evaluation for the initial prompt. Respond with each evaluation in between <EVALUATION> and </EVALUATION> as follows:

  1. <EVALUATION>Your Evaluation for initial prompt</EVALUATION>
  2. <EVALUATION>Your Evaluation for previous prompt 1</EVALUATION>
  ...
  n. <EVALUATION>Your Evaluation for previous prompt n</EVALUATION>

  """
          
          text_prompts = prompt.split("{image_placeholder}")

          user_content = [{"type": "text", "text": text_prompts[0]}]
          base64_images = [
              {
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{image}",
                      "detail": "high",
                  },
              }
          ]
          user_content.extend(base64_images)
          user_content.append({"type": "text", "text": text_prompts[1]})
          messages_template = [{"role": "user", "content": user_content}]

          return messages_template
          
      def generate_response(self, user_prompt, generated_image, previous_prompts=None):
          filtered_questions, VS_result = self.calculate_VS_score(user_prompt, generated_image)
          formatted_prompt = self.format_prompt_to_message(
                  user_prompt=user_prompt,
                  generated_image=generated_image,
                  previous_prompts=previous_prompts,
                  vs_result=VS_result,
              )
          print("Generating evaluation response...")
          response = generate_image_chat_response(formatted_prompt, client)
          return filtered_questions, VS_result, response
      
  prompt = "One full pitcher of beer with an elephant's trunk in it."
  reviewer = Reviewer()
  image = generate_image_robust(prompt)
  filtered_questions, VS_result, reviewer_evaluation = reviewer.generate_response(prompt, image)
  reviewer_evaluation

  print(reviewer_evaluation)

  class Challenger:
      """
      Agent B: 质疑者 (Challenger)
      - 负责对已给出的解进行“质疑”或“攻击”，找出潜在漏洞、不满足约束之处；
      - 可能提出改进思路，或抛出新的反例/约束来检验当前解。
      """
          
      def __init__(self):
          super().__init__()
          print("\nChallenger initialized.")
          print("----------------------")
          
      def format_prompt_to_message(
          self, user_prompt, previous_prompts, generated_image, vs_result, reviewer_evaluation
      ):
          image = encode_image_from_PIL_image(generated_image)

          VS_results = []
          for i, (key, value) in enumerate(vs_result["question_details"].items()):
              VS_result = "Element " + str(i) + "\n"
              VS_result += "Question: " + key + "\n"
              VS_result += "Ground Truth: " + value["answer"] + "\n"
              VS_result += (
                  "In the image generated from above prompt, the VQA model identified infer that the answer to the question is: "
                  + value["free_form_vqa"]
                  + "\n"
              )

              VS_results.append(VS_result)

          VS_results = "\n".join(VS_results)

          prompt = f"""
  You are a prompt challenger for text-to-image models. Your role is to critically evaluate the initial human-written prompt and previous prompts, identifying potential flaws and constraints that are not met based on the evaluation of the reviewer. Consider the scores assigned to each visual element in the outputs, with 1 indicating a perfect match and 0 indicating no match.

  Your task is to challenge the initial prompt: "{user_prompt}". Additionally, provide a critique of the previous prompts given.

  Here is the image that the text-to-image model generated based on the initial prompt:
  {{image_placeholder}}

  Here are the previous prompts and their visual element scores:
  ## Previous Prompts
  {previous_prompts}
  ## Visual Element Scores
  {VS_results}
  ## Reviewer's Evaluation
  {reviewer_evaluation}

  Based on the correctness and completeness of each prompt in relation to the generated images, identify potential weaknesses and unmet constraints. Propose improvement ideas or introduce new counterexamples and constraints to test the current solutions.
  If there are no previous prompts, focus on challenging the initial prompt. Respond with each challenge in between <CHALLENGE> and </CHALLENGE> as follows:

  1. <CHALLENGE>Your Challenge for initial prompt</CHALLENGE>
  2. <CHALLENGE>Your Challenge for previous prompt 1</CHALLENGE>
  ...
  n. <CHALLENGE>Your Challenge for previous prompt n</CHALLENGE>

  """
          # print(prompt)
          text_prompts = prompt.split("{image_placeholder}")

          user_content = [{"type": "text", "text": text_prompts[0]}]
          base64_images = [
              {
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{image}",
                      "detail": "high",
                  },
              }
          ]
          user_content.extend(base64_images)
          user_content.append({"type": "text", "text": text_prompts[1]})
          messages_template = [{"role": "user", "content": user_content}]

          return messages_template
      
      def generate_response(self, user_prompt, generated_image, filtered_questions, VS_result, reviewer_evaluation, previous_prompts=None):
          formatted_prompt = self.format_prompt_to_message(
                  user_prompt=user_prompt,
                  generated_image=generated_image,
                  previous_prompts=previous_prompts,
                  vs_result=VS_result,
                  reviewer_evaluation=reviewer_evaluation
              )
              
          print("Generating challenge response...")
          response = generate_image_chat_response(formatted_prompt, client)
          return response
      

  challenger = Challenger()
  challenger_response = challenger.generate_response(prompt, image, filtered_questions, VS_result, reviewer_evaluation)

  challenger_response

  class Refiner:
      '''
      Agent C: 修正者 (Refiner / Fixer)
      - 收到来自审阅者、质疑者的反馈后，对当前解进行修改、修补、重构；
      - 目标是提高解的质量，使之更符合目标需求或约束。
      '''

      def __init__(self):
          super().__init__()
          print("\nRefiner initialized.")
          print("----------------------")

      def format_prompt_to_message(
          self, user_prompt, previous_prompts, generated_image, vs_result, reviewer_evaluation, challenger_response
      ):
          image = encode_image_from_PIL_image(generated_image)

          VS_results = []
          for i, (key, value) in enumerate(vs_result["question_details"].items()):
              VS_result = "Element " + str(i) + "\n"
              VS_result += "Question: " + key + "\n"
              VS_result += "Ground Truth: " + value["answer"] + "\n"
              VS_result += (
                  "In the image generated from above prompt, the VQA model identified infer that the answer to the question is: "
                  + value["free_form_vqa"]
                  + "\n"
              )

              VS_results.append(VS_result)

          VS_results = "\n".join(VS_results)

          prompt = f"""
  You are a prompt refiner for text-to-image models. Your role is to improve the quality of the initial human-written prompt and previous prompts by incorporating feedback received from the reviewer and challenger. Your goal is to adjust, refine, and reconstruct the prompts to better meet the intended requirements and constraints.

  Your task is to refine the initial prompt: "{user_prompt}" and the previous prompts based on the feedback received.

  Here is the image that the text-to-image model generated based on the initial prompt:
  {{image_placeholder}}

  Here are the previous prompts and their visual element scores:
  ## Previous Prompts
  {previous_prompts}
  ## Visual Element Scores
  {VS_results}
  ## Reviewer's Evaluation
  {reviewer_evaluation}
  ## Challenger's Challenge
  {challenger_response}

  Using the feedback from both the reviewer and the challenger, modify and enhance the prompts to address weaknesses and fulfill unmet constraints. Generate improved prompts that capture the intended visual elements more effectively.
  If there are no previous prompts, focus on refining the initial prompt. Respond with each refined prompt in between <REFINED_PROMPT> and </REFINED_PROMPT> as follows:

  <REFINED_PROMPT>Your Refined prompt</REFINED_PROMPT>
  """
          # print(prompt)
          text_prompts = prompt.split("{image_placeholder}")

          user_content = [{"type": "text", "text": text_prompts[0]}]
          base64_images = [
              {
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{image}",
                      "detail": "high",
                  },
              }
          ]
          user_content.extend(base64_images)
          user_content.append({"type": "text", "text": text_prompts[1]})
          messages_template = [{"role": "user", "content": user_content}]

          return messages_template
      
      def generate_response(self, user_prompt, generated_image, filtered_questions, VS_result, reviewer_evaluation, challenger_response, previous_prompts=None):
          formatted_prompt = self.format_prompt_to_message(
                  user_prompt=user_prompt,
                  generated_image=generated_image,
                  previous_prompts=previous_prompts,
                  vs_result=VS_result,
                  reviewer_evaluation=reviewer_evaluation,
                  challenger_response=challenger_response
              )
              
          print("Generating refiner response...")
          response = generate_image_chat_response(formatted_prompt, client)
          return response
      
  refiner = Refiner()
  refiner_response = refiner.generate_response(prompt, image, filtered_questions, VS_result, reviewer_evaluation, challenger_response)

  def extract_refine_prompts(text):
      pattern = r"<REFINED_PROMPT>(.*?)</REFINED_PROMPT>"
      prompts = re.findall(pattern, text)
      return prompts[0]

  refine_prompts = extract_refine_prompts(refiner_response)
  refine_prompts

  new_image = generate_image_robust(refine_prompts)

  new_VS_result = get_VS_result(prompt, new_image, filtered_questions)

  new_VS_result

  def choose_best_image(image, new_image, VS_result, new_VS_result):
      if new_VS_result["VS_score"] > VS_result["VS_score"]:
          return new_image
      else:
          return image

  best_image = choose_best_image(image, new_image, VS_result, new_VS_result).save("best_image.jpg")