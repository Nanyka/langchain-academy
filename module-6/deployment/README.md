# Module 6 Deployment

This folder contains the LangGraph deployment for `task_maistro`.

## Local development

1. Create a local env file:

```bash
cp .env.example .env
```

2. Fill in the required values in `.env`.

At minimum:

- `IMAGE_NAME`
- `LANGSMITH_API_KEY`

3. Build the app image:

```bash
langgraph build -t my-image
```

If you changed `IMAGE_NAME` in `.env`, use the same tag here.

4. Start the deployment:

```bash
docker compose -f docker-compose-example.yml up
```

This starts three containers:

- `langgraph-api`
- `langgraph-redis`
- `langgraph-postgres`

## After code changes

If you change application code such as `task_maistro.py`, rebuild the image and restart the deployment:

```bash
langgraph build -t my-image
docker compose -f docker-compose-example.yml down
docker compose -f docker-compose-example.yml up
```

## Production-oriented workflow

For a real product, prefer immutable image tags instead of rebuilding an unversioned image like `my-image`.

Example:

```bash
langgraph build -t my-image:2026-03-22-1
```

Then set:

```env
IMAGE_NAME=my-image:2026-03-22-1
```

and deploy using that exact tag.

This gives you:

- traceable releases
- easier rollbacks
- clearer separation between code and runtime config

## Config

Runtime configuration is read from `docker-compose-example.yml` via environment variables.

Use `.env.example` as the template and keep your real `.env` local.
