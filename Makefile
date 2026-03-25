CLAUDE_MD    ?= $(HOME)/CLAUDE.md
COMPLETIONS  ?= $(HOME)/.local/share/zsh/site-functions

.PHONY: install uninstall

install:
	@uv pip install -e .
	@mkdir -p $(COMPLETIONS)
	@cp completions/_rex $(COMPLETIONS)/_rex
	@echo 'Installed rex completions to $(COMPLETIONS)/_rex'
	@if grep -q 'rex:start' $(CLAUDE_MD) 2>/dev/null; then \
		sed '/rex:start/,/rex:end/d' $(CLAUDE_MD) > $(CLAUDE_MD).tmp && \
		mv $(CLAUDE_MD).tmp $(CLAUDE_MD); \
	fi
	@cat CLAUDE.md >> $(CLAUDE_MD)
	@echo 'Updated rex instructions in $(CLAUDE_MD)'

uninstall:
	@uv pip uninstall rex 2>/dev/null || true
	@rm -f $(COMPLETIONS)/_rex
	@if [ -f $(CLAUDE_MD) ]; then \
		sed '/rex:start/,/rex:end/d' $(CLAUDE_MD) > $(CLAUDE_MD).tmp && \
		mv $(CLAUDE_MD).tmp $(CLAUDE_MD); \
	fi
	@echo 'rex uninstalled.'
