(type -p wget >/dev/null || (gapt update && sudo apt-get install wget -y)) \
	&& gmkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && cat $out | gtee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& gchmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | gtee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& gapt update \
	&& gapt install gh -y