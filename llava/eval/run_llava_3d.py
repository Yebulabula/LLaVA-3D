import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    process_videos,
    tokenizer_special_token,
    get_model_name_from_path,
)

import pandas as pd

from PIL import Image

import requests
from io import BytesIO
import re
from word2number import w2n
import json

import warnings
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*were not used.*")
warnings.filterwarnings("ignore", message=".*This IS expected if you are initializing.*")

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, filename):
    with open (filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def convert_words_to_digits(text):
    words = text.split()
    converted_words = []
    for word in words:
        try:
            # Attempt to convert the word to a number
            number = w2n.word_to_num(word)
            converted_words.append(str(number))
        except ValueError:
            # If the word is not a number, keep it as is
            converted_words.append(word)
    return ' '.join(converted_words)

def normalize_text(text):
    # Convert to lowercase and remove punctuation except digits and letters
    text = text.replace('To the', '').lower()
    if text.startswith('zero') or text.startswith('one') or text.startswith('two') or text.startswith('three') or text.startswith('four') or text.startswith('five') or text.startswith('six') or text.startswith('seven') or text.startswith('eight') or text.startswith('nine'):
        text = text.split(' ')[0]
        
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    
    # Strip extra whitespace
    text = text.strip()
    
    # Convert words to digits (e.g., "one" to "1")
    text = convert_words_to_digits(text)
    
    # Remove articles (optional, depending on context)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    
    return text

def partial_match_score(predicted, reference):
    pred_tokens = predicted.split()
    ref_tokens = reference.split()
    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    return len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0

def extract_non_nan_values(df):
    extracted_values = {}

    # Iterate over each row and extract the values
    for index, row in df.iterrows():
        scene_id = row['scene_id']
        if pd.notna(scene_id):
            # Create a dictionary to hold the non-NaN values for the current scene_id
            scene_values = {}

            # Extract non-NaN values for 'Front', 'Back', 'Left', 'Right'
            for direction in ['Front', 'Back', 'Left', 'Right']:
                value = row[direction]
                if pd.notna(value):
                    scene_values[direction] = value

            # Add the non-NaN values for the current scene_id to the overall dictionary
            if scene_values:
                extracted_values[scene_id] = scene_values

    return extracted_values

def eval_model(args):
    # Model
    disable_torch_init()
    
    template = '''
    Given a 3D scene, mentally rotate the image to align with the specified orientation.

    Scene Orientation: {}

    Now, given a context change, imagine how the scene would look after the change has been applied. Then, answer a question based on the changed scene.

    Context Change: {}
    Question: {}

    The answer should be a single word or short phrase.

    The answer is:
    '''

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    mode = 'video'
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, torch_dtype=torch_dtype
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "3D" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    eval_data = load_json(args.filename)
    
    # Metrics initialization per question type
    total_questions = 0
    exact_matches = 0
    partial_match_scores = []
    total_questions_per_type = {}
    exact_matches_per_type = {}
    partial_match_scores_per_type = {}
    
    df = pd.read_excel("dataset/Axis Definition.xlsx", sheet_name='Sheet1', engine='openpyxl')
    for scene_id, changes_list in eval_data.items():
        video_path = f'dataset/3D_VLM_data/{scene_id}'
        
        scene_orientation = extract_non_nan_values(df[df['scene_id'] == scene_id])
        scene_orientation = " ".join(
            f"The {item} was located at the {direction.lower()} of the scene."
            for scene_id, directions in scene_orientation.items()
            for direction, item in directions.items()
        )
        for changes in changes_list:
            context_change = changes['context_change']
            question_answers = changes['questions_answers']
            
            if mode == 'video':
                videos_dict = process_videos(
                    video_path,
                    processor['video'],
                    mode='random',
                    device=model.device
                )
                images_tensor = videos_dict['images'].to(model.device, dtype=torch_dtype)
                depths_tensor = videos_dict['depths'].to(model.device, dtype=torch_dtype)
                poses_tensor = videos_dict['poses'].to(model.device, dtype=torch_dtype)
                intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch_dtype)
            
            for qa in question_answers:
                question_type = qa['question_type']
                question = qa['question']
                reference_answer = qa['answer']
                    
                qs = template.format(scene_orientation, context_change, question)

                matches = re.search(r"\[([^\]]+)\]", qs)
                if matches:
                    coord_list = [float(x) for x in matches.group(1).split(',')]
                    coord_list = [round(coord, 3) for coord in coord_list[:3]]
                    qs = re.sub(r"\[([^\]]+)\]", "<boxes>", qs)
                    clicks = torch.tensor([coord_list])
                else:
                    clicks = torch.zeros((0,3))

                clicks_tensor = clicks.to(model.device, dtype=torch.bfloat16)
                
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if IMAGE_PLACEHOLDER in qs:
                    if model.config.mm_use_im_start_end:
                        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                    else:
                        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                else:
                    if model.config.mm_use_im_start_end:
                        qs = image_token_se + "\n" + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                        
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                input_ids = (
                    tokenizer_special_token(prompt, tokenizer, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        depths=depths_tensor,
                        poses=poses_tensor,
                        intrinsics=intrinsics_tensor,
                        clicks=clicks_tensor,
                        image_sizes=None,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

                predicted_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                qa['predicted_answer'] = predicted_answer
                
                predicted_answer = normalize_text(predicted_answer)
                reference_answer = normalize_text(reference_answer)
                
                print(f"predicted_answer: {predicted_answer}")
                print(f"reference_answer: {reference_answer}")
                
                # Initialize metrics for new question types
                if question_type not in total_questions_per_type:
                    total_questions_per_type[question_type] = 0
                    exact_matches_per_type[question_type] = 0
                    partial_match_scores_per_type[question_type] = []

                # Exact Match
                if predicted_answer == reference_answer:
                    exact_matches += 1
                    exact_matches_per_type[question_type] += 1

                # Partial Match Score
                partial_match = partial_match_score(predicted_answer, reference_answer)
                partial_match_scores.append(partial_match)
                partial_match_scores_per_type[question_type].append(partial_match)

                total_questions += 1
                total_questions_per_type[question_type] += 1
    
    save_json(eval_data, f"dataset/context_vqa_noC_with_context_change_{args.model_path.split('/')[-1]}_no_label_rotated.json")
    
    # Calculate average metrics for each question type
    for question_type in total_questions_per_type:
        exact_match_score_per_type = (exact_matches_per_type[question_type] / total_questions_per_type[question_type]) * 100
        average_partial_match_per_type = sum(partial_match_scores_per_type[question_type]) / len(partial_match_scores_per_type[question_type]) * 100
        
        # Print results for each question type
        print(f"Question Type: {question_type}")
        print(f"  Exact Match Score: {exact_match_score_per_type:.2f}%")
        print(f"  Partial Match Score: {average_partial_match_per_type:.2f}%")


    # Calculate overall average metrics
    exact_match_score = (exact_matches / total_questions) * 100
    average_partial_match_score = sum(partial_match_scores) / len(partial_match_scores) * 100

    # Print overall results
    print("\nOverall Metrics:")
    print(f"Exact Match Score: {exact_match_score:.2f}%")
    print(f"Partial Match Score: {average_partial_match_score:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--video-path", type=str, help="Path to the video file")
    # group.add_argument("--image-file", type=str, help="Path to the image file")
    parser.add_argument(
        "-f", "--filename",
        type=str,
        required=True,
        help="The name of the file to process"
    )
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
