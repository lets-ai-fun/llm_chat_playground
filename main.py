from langchain.globals import set_debug

import user_ui
import chain_api

#enable llm debug log -> set to False or comment line if you don't want this log(s)
set_debug(True)

chain, chain_type = user_ui.start_program_ui()
chain_api.endless_chat(chain= chain, chain_type= chain_type)

