from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import re

import utils.prompts as prompts
import utils.models as models

# run judge on pairs
# judges should take a question and two responses, and return a decision (e.g., A>B or B>A)
# Note B>A and not A<B
# Judge.get_judgment must be async!
# For a new judge, add a corresponding entry to get_judge_from_judge_name_and_model


class Judge(ABC):
    """Abstract base class for all judge implementations."""

    @abstractmethod
    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get a judgment for a pair of answers given a question.

        Parameters
        ----------
        question : str
            The prompt or question presented to the models.
        answer_A : str
            The response from model A.
        answer_B : str
            The response from model B.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the 'judgment' (raw text) and 'decision' (e.g., 'A>B').
        """
        pass


class ArenaHard(Judge):
    """Judge implementation following the Arena-Hard methodology.

    See: https://github.com/lmarena/arena-hard-auto/blob/4ce0f0087776158a4461162cbef1d9bb5464bb57/gen_judgment.py
    """

    def __init__(self, model_name: str):
        """Initialize the ArenaHard judge.


        Parameters
        ----------
        model_name : str
            The name of the judge model to use.
        """
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.number_of_judgment_attempts = 2

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using the Arena-Hard prompt template.

        Parameters
        ----------
        question : str
            The original prompt.
        answer_A : str
            First model's response.
        answer_B : str
            Second model's response.

        Returns
        -------
        Dict[str, Any]
            The result of the judgment, including the raw response and the final decision.
        """
        system_message = prompts.render_template("arena_hard_judge_system")
        user_message = prompts.render_template(
            "arena_hard_judge_prompt",
            prompt=question,
            answer_a=answer_A,
            answer_b=answer_B,
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        judgment = ""
        for _ in range(self.number_of_judgment_attempts):
            new_judgment = await self.api.chat(
                messages=messages,
                temperature=0,
                max_tokens=4096,
            )
            judgment += "\n" + new_judgment
            score, try_again = self.get_score(
                judgment, re.compile("\[\[([AB<>=]+)\]\]")
            )
            messages.append({"role": "assistant", "content": new_judgment})
            if not try_again:
                break
            messages.append(
                {
                    "role": "user",
                    "content": "continue your judgment and finish by outputting a final verdict label",
                }
            )
        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": messages[1]["content"],
                "response": judgment,
            },
            "decision": score.replace(">>", ">").strip() if score else None,
        }

    def get_score(
        self, judgment: str, pattern: re.Pattern, pairwise: bool = True
    ) -> Tuple[Union[int, str, None], Optional[bool]]:
        """Extract the score or decision from the judge's response text.

                Parameters
                ----------
                judgment : str
                    The raw text from the judge model.
                pattern : re.Pattern
                    The regex pattern to search for in the judgment text.
                pairwise : bool, optional
                    Whether this is a pairwise comparison (default is True).

                Returns
        -------
                Tuple[Union[int, str, None], Optional[bool]]
                    The extracted score/decision and a boolean indicating if the extraction failed (try again).
        """
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None, True
        elif len(set(matches)) == 1:
            if pairwise:
                return matches[0].strip("\n"), False
            return int(matches[0]), False
        else:
            return None, False


class Vanilla(Judge):
    """A basic judge implementation with minimal formatting."""

    def __init__(self, model_name: str) -> None:
        """Initialize the Vanilla judge.

        Parameters
        ----------
        model_name : str
            The name of the judge model.
        """
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    def extract_pairwise_result(self, raw_output: str) -> str:
        """Extract the pairwise result from the raw model output.

        Parameters
        ----------
        raw_output : str
            The raw text from the model.

        Returns
        -------
        str
            The formatted decision (e.g., 'A>B').

        Raises
        ------
        Exception
            If the output cannot be parsed.
        """
        print("raw:", raw_output)
        if raw_output == "Output (a)":
            return "A>B"
        elif raw_output == "Output (b)":
            return "B>A"
        raise Exception("Cannot parse output:", raw_output)

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment for the answer pair.

        Parameters
        ----------
        question : str
            The original question.
        answer_A : str
            Model A's response.
        answer_B : str
            Model B's response.

        Returns
        -------
        Dict[str, Any]
            The judgment result and decision.
        """
        prompt = prompts.render_template(
            "vanilla_prompt", question=question, answer_a=answer_A, answer_b=answer_B
        )
        print("prompt:", prompt)
        output = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )

        pred_label = self.extract_pairwise_result(output.strip())

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": pred_label,
        }


class PandaLM(Judge):
    """Judge implementation using PandaLM.

    See: https://github.com/WeOpenML/PandaLM
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the PandaLM judge.

        Parameters
        ----------
        model_name : str
            The name of the PandaLM model.
        """
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.pattern = re.compile(
            r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name, use_fast=False
        )

    def truncate_responses(
        self,
        question: str,
        answer_A: str,
        answer_B: str,
        context_limit: int,
        max_new_tokens: int,
        truncation_side: str,
    ) -> Tuple[str, str]:
        """Truncate responses to fit within the context limit.

        Parameters
        ----------
        question : str
            The prompt or question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.
        context_limit : int
            Total token limit for the context.
        max_new_tokens : int
            Number of tokens reserved for the output.
        truncation_side : str
            Which side to truncate ('left' or 'right').

        Returns
        -------
        Tuple[str, str]
            Truncated versions of answer_A and answer_B.
        """
        template_with_question = prompts.render_template(
            "pandalm_prompt", instruction=question, resp1="", resp2=""
        )
        len_template = len(
            self.tokenizer(template_with_question).input_ids
        )  # includes special BOS token <s>
        tokens_per_response = (
            (context_limit - max_new_tokens - len_template) // 2 - 2
        )  # each response should be truncated to a length of tokens_per_response

        answer_A_tokenized = self.tokenizer(
            answer_A,
            add_special_tokens=False,  # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_A_tokenized_truncated = (
            answer_A_tokenized[:tokens_per_response]
            if truncation_side == "right"
            else answer_A_tokenized[-tokens_per_response:]
        )  # left
        answer_A_truncated = self.tokenizer.decode(
            answer_A_tokenized_truncated
        )  # should not be any special tokens anyways

        answer_B_tokenized = self.tokenizer(
            answer_B,
            add_special_tokens=False,  # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_B_tokenized_truncated = (
            answer_B_tokenized[:tokens_per_response]
            if truncation_side == "right"
            else answer_B_tokenized[-tokens_per_response:]
        )  # left
        answer_B_truncated = self.tokenizer.decode(
            answer_B_tokenized_truncated
        )  # should not be any special tokens anyways

        return answer_A_truncated, answer_B_truncated

    def build_pandalm_prompt(self, instruction: str, resp1: str, resp2: str) -> str:
        """Construct the prompt for PandaLM.

        Parameters
        ----------
        instruction : str
            The instruction/question.
        resp1 : str
            First response.
        resp2 : str
            Second response.

        Returns
        -------
        str
            The rendered prompt.
        """
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        input_sequence = prompts.render_template(
            "pandalm_prompt", instruction=instruction, resp1=resp1, resp2=resp2
        )
        return input_sequence + "\n"  # why does jinja strip the training new line?

    def parse_pandalm_response(self, text: str) -> int:
        """Parse the response from PandaLM to extract the decision.

        Parameters
        ----------
        text : str
            The raw output from the model.

        Returns
        -------
        int
            The numerical decision (1 for A, 2 for B, 0 for tie).
        """
        sp = text.strip().split("\n")
        if sp[0] in ["1", "2"]:
            return int(sp[0])
        elif sp[0].lower() == "tie":
            return 0
        else:
            return 0

    def postprocess_output(self, text: str) -> str:
        """Postprocess the PandaLM output text.

        Parameters
        ----------
        text : str
            Raw output text.

        Returns
        -------
        str
            Cleaned output text.
        """
        text = text.strip()
        self.pattern.sub("", text.strip()).strip()
        return text

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Perform judgment using PandaLM.

        Parameters
        ----------
        question : str
            The prompt or question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The final judgment and decision.
        """
        answer_A, answer_B = self.truncate_responses(
            question,
            answer_A,
            answer_B,
            context_limit=2048,
            max_new_tokens=150,  # we only need the first few tokens to determine decision
            truncation_side="left",
        )

        prompt = self.build_pandalm_prompt(
            instruction=question,
            resp1=answer_A,
            resp2=answer_B,
        )

        output = await self.api.complete(
            prompt=prompt,
            temperature=0,
            top_p=1,
            max_tokens=150,
            extra_body={
                "use_beam_search": True,
                "best_of": 4,
                "early_stopping": True,
                "repetition_penalty": 1.2,
            },
        )

        resp = self.postprocess_output(output)
        out = self.parse_pandalm_response(resp)
        if out == 1:
            decision = "A>B"
        elif out == 2:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": resp,
            },
            "decision": decision,
        }


class JudgeLM(Judge):
    """Judge implementation using JudgeLM.

    See: https://github.com/baaivision/JudgeLM
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the JudgeLM judge.

        Parameters
        ----------
        model_name : str
            The name of the JudgeLM model.
        """
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name, use_fast=False
        )

    def truncate_responses(
        self,
        question: str,
        answer_A: str,
        answer_B: str,
        context_limit: int,
        max_new_tokens: int,
        truncation_side: str,
    ) -> Tuple[str, str]:
        """Truncate responses for JudgeLM.

        Parameters
        ----------
        question : str
            The prompt or question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.
        context_limit : int
            Context token limit.
        max_new_tokens : int
            Output token reservation.
        truncation_side : str
            Truncation side ('left' or 'right').

        Returns
        -------
        Tuple[str, str]
            Truncated responses.
        """
        template_with_question = prompts.render_template(
            "judgelm_prompt", question=question, answer_1="", answer_2=""
        )
        len_template = len(
            self.tokenizer(template_with_question).input_ids
        )  # includes special BOS token <s>
        tokens_per_response = (
            (context_limit - max_new_tokens - len_template) // 2 - 2
        )  # each response should be truncated to a length of tokens_per_response

        answer_A_tokenized = self.tokenizer(
            answer_A,
            add_special_tokens=False,  # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_A_tokenized_truncated = (
            answer_A_tokenized[:tokens_per_response]
            if truncation_side == "right"
            else answer_A_tokenized[-tokens_per_response:]
        )  # left
        answer_A_truncated = self.tokenizer.decode(
            answer_A_tokenized_truncated
        )  # should not be any special tokens anyways

        answer_B_tokenized = self.tokenizer(
            answer_B,
            add_special_tokens=False,  # we dont want to include the BOS token here
            padding=False,
            truncation=False,
        ).input_ids
        answer_B_tokenized_truncated = (
            answer_B_tokenized[:tokens_per_response]
            if truncation_side == "right"
            else answer_B_tokenized[-tokens_per_response:]
        )  # left
        answer_B_truncated = self.tokenizer.decode(
            answer_B_tokenized_truncated
        )  # should not be any special tokens anyways

        return answer_A_truncated, answer_B_truncated

    def parse_score(self, review: str) -> List[float]:
        """Parse the scores from the judge's review text.

                Parameters
                ----------
                review : str
                    The review text from the judge.

                Returns
        -------
                List[float]
                    A list containing the scores for model A and model B.
        """
        try:
            score_pair = review.split("\n")[0]
            score_pair = score_pair.replace(",", " ")
            sp = score_pair.split(" ")
            if len(sp) == 2:
                return [float(sp[0]), float(sp[1])]
            else:
                raise Exception()
        except Exception:
            return [-1.0, -1.0]

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using JudgeLM.

        Parameters
        ----------
        question : str
            The question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The judgment result and decision.
        """
        answer_A, answer_B = self.truncate_responses(
            question,
            answer_A,
            answer_B,
            context_limit=2048,
            max_new_tokens=16,
            truncation_side="right",
        )

        prompt = prompts.render_template(
            "judgelm_prompt", question=question, answer_1=answer_A, answer_2=answer_B
        )

        output = await self.api.complete(
            prompt=prompt,
            temperature=0.0,  # https://github.com/baaivision/JudgeLM/blob/ce12b12779764fe06e28c797cecee86018a298e4/judgelm/llm_judge/gen_model_judgement_multi.py#L235
            max_tokens=16,
        )

        scores = self.parse_score(output)

        if scores[0] > scores[1]:
            decision = "A>B"
        elif scores[0] < scores[1]:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision,
        }


class AutoJ(Judge):
    """Judge implementation using Auto-J.

    See: https://github.com/GAIR-NLP/auto-j
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the Auto-J judge.

        Parameters
        ----------
        model_name : str
            The name of the Auto-J model.
        """
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    def extract_pariwise_result(self, raw_output: str) -> Optional[str]:
        """Extract the pairwise result from Auto-J's output.

        Parameters
        ----------
        raw_output : str
            The raw text from the model.

        Returns
        -------
        Optional[str]
            The decision ('A>B', 'B>A', 'A=B') or None if not found.
        """
        raw_output = raw_output.strip()
        pos = raw_output.rfind("final decision is ")
        pred_label = None
        if pos != -1:
            pred_rest = raw_output[pos + len("final decision is ") :].strip().lower()
            if pred_rest.startswith("response 1"):
                pred_label = "A>B"
            elif pred_rest.startswith("response 2"):
                pred_label = "B>A"
            elif pred_rest.startswith("tie"):
                pred_label = "A=B"
        return pred_label

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using Auto-J.

        Parameters
        ----------
        question : str
            The question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The judgment result and decision.
        """
        prompt = prompts.render_template(
            "autoj_prompt",
            question=question,
            response=answer_A,
            response_another=answer_B,
        )

        output = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )  # SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024) https://github.com/GAIR-NLP/auto-j

        pred_label = self.extract_pariwise_result(output)

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": pred_label,
        }


class Prometheus2(Judge):
    """Judge implementation using Prometheus 2.

    See: https://github.com/prometheus-eval/prometheus-eval
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the Prometheus2 judge.

        Parameters
        ----------
        model_name : str
            The name of the Prometheus model.
        """
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)
        self.rubric = "[Are the model's responses factually correct and well-supported by evidence?]"  # https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/prompts.py
        self.REL_SYSTEM_PROMPT = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

    def _parse_output_relative(
        self, output: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse the relative output from Prometheus 2.

        Parameters
        ----------
        output : str
            The raw text output from the model.

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            A tuple of (full output, extracted result 'A' or 'B').
        """
        explicit_pattern = r"""
            (?:                                # Start of non-capturing group
                \[RESULT\]|\[RESULT:\s*|        # Match [RESULT] or [RESULT:
                \[Response\s+|                  # Match [Response
                # Match [Result] or [Result] Response
                \[Result\](?:\s+Response)?|
                \[Result:\s*|                   # Match [Result:
                # Match Result: at the start of a line
                (?:^|\n)Result:?\s*
            )                                   # End of non-capturing group
            \s*                                 # Allow any whitespace
            (A|B)                               # Capture A or B
            (?:\]|\s|$)                         # Allow closing bracket, whitespace, or end of string
        """
        match = re.search(
            explicit_pattern, output, re.IGNORECASE | re.VERBOSE | re.MULTILINE
        )

        if match:
            result = match.group(1).upper()
            return output, result

        return None, None

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using Prometheus 2.

        Parameters
        ----------
        question : str
            The original instruction.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The judgment and decision.
        """
        prompt = prompts.render_template(
            "prometheus2_prompt",
            instruction=question,
            response_A=answer_A,
            response_B=answer_B,
            rubric=self.rubric,
        )

        messages = [
            {"role": "system", "content": self.REL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )  # https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/utils.py

        _, scores = self._parse_output_relative(output)

        decision = None  # no tie option
        if scores == "A":
            decision = "A>B"
        elif scores == "B":
            decision = "B>A"

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision,
        }


class SkyworkCritic(Judge):
    """Judge implementation using Skywork-Critic."""

    def __init__(self, model_name: str) -> None:
        """Initialize the SkyworkCritic judge.

        Parameters
        ----------
        model_name : str
            The name of the Skywork-Critic model.
        """
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using Skywork-Critic.

        Parameters
        ----------
        question : str
            The question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The judgment result and decision.
        """
        prompt = prompts.render_template(
            "skywork_critic_prompt",
            input=question,
            response_a=answer_A,
            response_b=answer_B,
        )

        messages = [{"role": "user", "content": prompt}]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )

        if "A" in output:
            decision = "A>B"
        elif "B" in output:
            decision = "B>A"
        else:
            decision = None

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": prompt,
                "response": output,
            },
            "decision": decision,
        }


class InternLM2Reward(Judge):
    """Judge implementation using InternLM2-Reward model."""

    def __init__(
        self, model_name: str = "internlm/internlm2-20b-reward", device: str = "cuda:0"
    ):
        """Initialize the InternLM2-Reward judge.

        Parameters
        ----------
        model_name : str, optional
            The name of the reward model (default is "internlm/internlm2-20b-reward").
        device : str, optional
            The device to run the model on (default is "cuda:0").
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.rm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment by comparing scores from InternLM2-Reward.

        Parameters
        ----------
        question : str
            The question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The scores and the resulting decision.
        """
        conv1 = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_A},
        ]
        conv2 = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_B},
        ]

        score1 = self.rm.get_score(self.rm_tokenizer, conv1)
        score2 = self.rm.get_score(self.rm_tokenizer, conv2)

        judgement = "A>B" if score1 > score2 else "B>A"

        return {
            "judgment": {"judge_model": self.model_name, "scores": [score1, score2]},
            "decision": judgement,
        }


class GRMReward(Judge):
    """Judge implementation using GRM-Reward model."""

    def __init__(
        self,
        model_name: str = "Ray2333/GRM-Gemma-2B-rewardmodel-ft",
        device: str = "cuda:0",
    ):
        """Initialize the GRM-Reward judge.

        Parameters
        ----------
        model_name : str, optional
            The reward model name (default is "Ray2333/GRM-Gemma-2B-rewardmodel-ft").
        device : str, optional
            The device to use (default is "cuda:0").
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

    def get_reward(self, message: List[Dict[str, str]]) -> float:
        """Calculate the reward score for a single conversation.

        Parameters
        ----------
        message : List[Dict[str, str]]
            The conversation messages.

        Returns
        -------
        float
            The calculated reward score.
        """
        import torch

        message_template = self.tokenizer.apply_chat_template(message, tokenize=False)

        kwargs = {"padding": "max_length", "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer.encode_plus(message_template, **kwargs)

        with torch.no_grad():
            reward_tensor = self.reward_model(
                tokens["input_ids"][0].view(1, -1).to(self.device),
                attention_mask=tokens["attention_mask"][0].view(1, -1).to(self.device),
            )[0]
            reward = reward_tensor.cpu().detach().item()

        return reward

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using GRM-Reward scores.

        Parameters
        ----------
        question : str
            The original question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The scores and decision.
        """
        message_A = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_A},
        ]
        message_B = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_B},
        ]

        score_A = self.get_reward(message_A)
        score_B = self.get_reward(message_B)

        judgement = "A>B" if score_A > score_B else "B>A"

        return {
            "judgment": {"judge_model": self.model_name, "scores": [score_A, score_B]},
            "decision": judgement,
        }


class SkyworkReward(Judge):
    """Judge implementation using Skywork-Reward model."""

    def __init__(self, model_name: str, device: str = "cuda:0"):
        """Initialize the Skywork-Reward judge.

        Parameters
        ----------
        model_name : str
            The Skywork reward model name.
        device : str, optional
            The device to use (default is "cuda:0").
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model_name = model_name
        self.device = device
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using Skywork-Reward scores.

        Parameters
        ----------
        question : str
            The question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The scores and decision.
        """
        import torch

        conv1 = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_A},
        ]
        conv2 = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_B},
        ]

        conv1_formatted = self.rm_tokenizer.apply_chat_template(conv1, tokenize=False)
        conv2_formatted = self.rm_tokenizer.apply_chat_template(conv2, tokenize=False)
        conv1_tokenized = self.rm_tokenizer(conv1_formatted, return_tensors="pt").to(
            self.device
        )
        conv2_tokenized = self.rm_tokenizer(conv2_formatted, return_tensors="pt").to(
            self.device
        )

        # Get the reward scores
        with torch.no_grad():
            score1 = self.rm(**conv1_tokenized).logits[0][0].item()
            score2 = self.rm(**conv2_tokenized).logits[0][0].item()

        judgement = "A>B" if score1 > score2 else "B>A"

        return {
            "judgment": {"judge_model": self.model_name, "scores": [score1, score2]},
            "decision": judgement,
        }


class CompassJudger(Judge):
    """Judge implementation using Compass Judger."""

    def __init__(self, model_name: str) -> None:
        """Initialize the CompassJudger.

        Parameters
        ----------
        model_name : str
            The name of the judge model.
        """
        self.model_name = model_name
        self.api = models.get_chat_api_from_model(model_name)

    def get_score(
        self, judgment: str, pattern: re.Pattern, pairwise: bool = True
    ) -> Tuple[Union[int, str, None], Optional[bool]]:
        """Extract the score or decision from the judgment text.

        Parameters
        ----------
        judgment : str
            Raw text from the judge.
        pattern : re.Pattern
            Regex pattern to extract decision.
        pairwise : bool, optional
            Whether it's a pairwise comparison (default is True).

        Returns
        -------
        Tuple[Union[int, str, None], Optional[bool]]
            The extracted score/decision and whether to retry.
        """
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None, True
        elif len(set(matches)) == 1:
            if pairwise:
                return matches[0].strip("\n"), False
            return int(matches[0]), False
        else:
            return None, False

    async def get_judgment(
        self, question: str, answer_A: str, answer_B: str
    ) -> Dict[str, Any]:
        """Get judgment using Compass Judger prompt.

        Parameters
        ----------
        question : str
            The question.
        answer_A : str
            Response A.
        answer_B : str
            Response B.

        Returns
        -------
        Dict[str, Any]
            The judgment result and decision.
        """
        system_message = prompts.render_template("arena_hard_judge_system")
        user_message = prompts.render_template(
            "arena_hard_judge_prompt",
            prompt=question,
            answer_a=answer_A,
            answer_b=answer_B,
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        output = await self.api.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )

        score, _ = self.get_score(output, re.compile("\[\[([AB<>=]+)\]\]"))

        return {
            "judgment": {
                "judge_model": self.model_name,
                "prompt": messages[1]["content"],
                "response": output,
            },
            "decision": score.replace(">>", ">").strip() if score else None,
        }


def get_judge_from_judge_name_and_model(judge_name: str, judge_model: str) -> Judge:
    """Factory function to get a judge instance by name and model.

    Parameters
    ----------
    judge_name : str
        The name of the judge type (e.g., 'arena_hard', 'vanilla', etc.).
    judge_model : str
        The model identifier for the judge.

    Returns
    -------
    Judge
        An instance of the specified Judge implementation.

    Raises
    ------
    NotImplementedError
        If the judge name or model is not supported.
    """
    if judge_name == "arena_hard":
        return ArenaHard(judge_model)
    elif judge_name == "vanilla":
        return Vanilla(judge_model)
    elif judge_name == "panda_lm":
        return PandaLM(judge_model)
    elif judge_name == "judge_lm":
        return JudgeLM(judge_model)
    elif judge_name == "auto_j":
        return AutoJ(judge_model)
    elif judge_name == "prometheus_2":
        return Prometheus2(judge_model)
    elif judge_name == "skywork_critic":
        return SkyworkCritic(judge_model)
    elif judge_name == "compass_judger":
        return CompassJudger(judge_model)
    elif judge_name == "reward_model":
        if judge_model in [
            "internlm/internlm2-7b-reward",
            "internlm/internlm2-20b-reward",
        ]:
            return InternLM2Reward(judge_model)
        elif judge_model in ["Ray2333/GRM-Gemma-2B-rewardmodel-ft"]:
            return GRMReward(judge_model)
        elif judge_model in [
            "Skywork/Skywork-Reward-Gemma-2-27B",
            "Skywork/Skywork-Reward-Llama-3.1-8B",
        ]:
            return SkyworkReward(judge_model)
        else:
            raise NotImplementedError(
                f"Judge with name {judge_name} for model with name {judge_model} is not yet implemented."
            )
    else:
        raise NotImplementedError(
            f"Judge with name {judge_name} for model with name {judge_model} is not yet implemented."
        )
