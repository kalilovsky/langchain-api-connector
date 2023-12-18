from PromptKernel.IntentionRouter.IntentionRouter import IntentionRouter
from PromptKernel.Types.UserPrompts import UserPrompts


class PromptKernel:
    user_prompt: UserPrompts

    def handle_prompt(self, user_prompt: UserPrompts):
        self.user_prompt = user_prompt
        router = IntentionRouter()
        return router.route(user_prompt)
