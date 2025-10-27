
.PHONY: cleanup
cleanup:
	uv tool run pre-commit run --all
