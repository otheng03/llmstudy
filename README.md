# llmstudy

# Setup

## Zed

### Create pyproject.toml

```
$ $PROJECT_HOME/llmstudy pyproject.toml
$ vi pyproject.toml
```

Type the content below, save pyproject.toml, quit vi
```
[tool.pyright]
venvPath = "."
venv = "venv"
```

Open zed
```
$ zed -n $PROJECT_HOME/llmstudy
```

### Install Ruff linter extension

`cmd + shift + p` -> `zed: extensions` -> Install Ruff

`cmd + shift + p` -> `zed: open project setting` -> Type the content below
```
{
  "languages": {
    "Python": {
      "language_servers": ["ruff"]
    }
  }
}
```
