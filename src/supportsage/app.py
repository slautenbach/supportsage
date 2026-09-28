from __future__ import annotations

import argparse

import gradio as gr

from supportsage.inference import SupportSageModel


def build_demo(model: SupportSageModel) -> gr.Blocks:
    def respond(message: str, history: list[dict[str, str]]) -> str:
        del history
        return model.generate(message)

    return gr.ChatInterface(
        fn=respond,
        title="SupportSage",
        description="A Qwen2.5-0.5B assistant fine-tuned with LoRA for customer-support responses.",
        examples=[
            "I need to cancel order {{Order Number}}.",
            "My package says delivered, but I cannot find it.",
            "How do I update the payment method on my account?",
        ],
        chatbot=gr.Chatbot(height=520, layout="bubble"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the SupportSage demo.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="artifacts/supportsage-qwen-lora")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    adapter = args.adapter if args.adapter else None
    build_demo(SupportSageModel(args.model, adapter)).launch(share=args.share)


if __name__ == "__main__":
    main()

