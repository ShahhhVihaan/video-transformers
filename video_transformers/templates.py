from pathlib import Path
from typing import Any, Dict, List


def generate_labels_table(labels: List[str]) -> str:
    str_1 = """
| Labels |
| :-- |
"""
    str_2 = "\n".join(["| " + label + " |" for label in labels])
    return str_1 + str_2


def generate_dict_to_table(_dict: Dict[str, Any]) -> str:
    if not _dict:
        return "N/A"
    else:
        return "\n".join([f"| {key} | {value} |" for key, value in _dict.items()])


def generate_hf_model_card(
    task: str = "single_label_classification",
    total_model_params: int = None,
    total_trainable_model_params: int = None,
    labels: List[str] = None,
    preprocessor_config: Dict[str, Any] = None,
    backbone_config: Dict[str, Any] = None,
    neck_config: Dict[str, Any] = None,
    head_config: Dict[str, Any] = None,
) -> str:
    return (
        f"""
---
library_name: video-transformers
tags:
- Video Transformers
- video-transformers
- video-classification
- image-classification
---

## Usage

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained(model_name_or_path)

model.predict(video_path="video.mp4")
>> [{{'filename': "video.mp4", 'predictions': {{'class1': 0.98, 'class2': 0.02}}}}]
```

- Refer to [video-transformers](https://github.com/video-transformers/) for more details.

## Model description

This model is intended to be used for the task of classifying videos. 
A video is an ordered sequence of frames. An individual frame of a video has spatial information whereas a sequence of video frames have temporal information.
This model can predict the following {len(labels)} labels:

{generate_labels_table(labels)}

| Model Details | Value |
| Task | {task} |
| Total Model Params | {str(total_model_params)} |
| Total Trainable Model Params | {str(total_trainable_model_params)} |

| Preprocessor Config | Value |
| :-- | :-- |
"""
        + generate_dict_to_table(preprocessor_config)
        + """

| Backbone Config | Value |
| :-- | :-- |
"""
        + generate_dict_to_table(backbone_config)
        + """
    
| Neck Config | Value |
| :-- | :-- |
"""
        + generate_dict_to_table(neck_config)
        + """

| Head Config | Value |
| :-- | :-- |
"""
        + generate_dict_to_table(head_config)
        + """

Model-card auto generated by [video-transformers](https://github/fcakyon/video-transformers).
"""
    )


def export_hf_model_card(
    export_dir: str,
    task: str = "single_label_classification",
    total_model_params: int = None,
    total_trainable_model_params: int = None,
    labels: List[str] = None,
    preprocessor_config: Dict[str, Any] = None,
    backbone_config: Dict[str, Any] = None,
    neck_config: Dict[str, Any] = None,
    head_config: Dict[str, Any] = None,
) -> str:
    export_path = Path(export_dir) / "README.md"
    # save as readme.md
    with open(export_path, "w") as f:
        f.write(
            generate_hf_model_card(
                task=task,
                total_model_params=total_model_params,
                total_trainable_model_params=total_trainable_model_params,
                labels=labels,
                preprocessor_config=preprocessor_config,
                backbone_config=backbone_config,
                neck_config=neck_config,
                head_config=head_config,
            )
        )


def generate_gradio_app(
    model_path_or_url: str,
    examples: List[str],
    author_username: str = None,
) -> str:
    from video_transformers import VideoModel

    model = VideoModel.from_pretrained(model_path_or_url)

    return f"""
import gradio as gr

from video_transformers import VideoModel

model: VideoModel = VideoModel.from_pretrained("{model_path_or_url}")

app = gr.Blocks()

with app:
    gr.Markdown("# **<p align='center'>Video Classification with Transformers</p>**")
    gr.Markdown("This space demonstrates the use of hybrid Transformer-based models for video classification.")
    gr.Markdown(f"The model is trained to classify videos belonging to the following classes: {model.labels}")

    with gr.Tabs():
        with gr.TabItem("Upload & Predict"):
            with gr.Box():

                with gr.Row():
                    input_video = gr.Video(label="Input Video", show_label=True)
                    output_label = gr.Label(label="Model Output", show_label=True)

            gr.Markdown("**Predict**")

            with gr.Box():
                with gr.Row():
                    submit_button = gr.Button("Submit")

            gr.Markdown("**Examples:**")

            # gr.Markdown("CricketShot, PlayingCello, Punch, ShavingBeard, TennisSwing")

            with gr.Column():
                gr.Examples({examples}, [input_video], [output_label], model.predict, cache_examples=True)

            submit_button.click(model.predict, inputs=input_video, outputs=[output_label])

            gr.Markdown("**Note:** The model is trained to classify videos belonging to the following classes: {model.labels}")

            gr.Markdown("**Credits:**")
            gr.Markdown("This space is powered by [video-transformers]('https://github.com/video-transformers/')")
            gr.Markdown("{"This model is finetuned by '" + author_username+"'." if author_username else ""}")

app.launch()
"""


if __name__ == "__main__":
    from video_transformers import VideoModel

    model: VideoModel = VideoModel.from_pretrained("runs/hf_exp15/checkpoint")
    print(
        generate_hf_model_card(
            labels=model.labels,
            backbone_config=model.config["backbone"],
            neck_config=model.config["neck"],
            preprocessor_config=model.preprocessor_config,
            head_config=model.config["head"],
            total_model_params=model.num_total_params,
            total_trainable_model_params=model.num_trainable_params,
        )
    )
    print(generate_gradio_app("runs/hf_exp15", examples=["video.mp4"], author_username="fcakyon"))
