# Deploy to AWS (ECS Express Mode + ECR + Bedrock)

> AWS App Runner stopped accepting new customers on 30 April 2026. AWS recommends **Amazon ECS Express Mode**:
> one command creates Fargate tasks, an Application Load Balancer with HTTPS, health checks, autoscaling and logs.

```
Internet ──HTTPS──> ALB (managed by Express Mode) ──> ECS Fargate tasks (this API, N replicas)
                                                         ├─> Amazon Bedrock (LLM + embeddings)  [or OpenAI]
                                                         └─> Chroma (local per task)  → production: OpenSearch / Aurora pgvector
```

## 0. Prerequisites
- An AWS account, AWS CLI v2 configured (`aws configure`), Docker running.
- **Costs:** a load balancer and Fargate tasks cost money every hour they run. Delete everything after the demo (step 6).
- Set variables (Linux/macOS/Git Bash):
```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO=plant-ops-copilot
```

## 1. Push the image to Amazon ECR
```bash
aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker build --platform linux/amd64 -t $REPO .
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

## 2. Create the two IAM roles (once per account)
```bash
aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam create-role --role-name ecsInfrastructureRoleForExpressServices --assume-role-policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
```
IAM is eventually consistent: if the next step says *Unable to assume the service linked role*, wait a minute and retry.

## 3. Choose the LLM in the cloud
- **Option A – Amazon Bedrock (recommended for the interview story):** enable a model in *Bedrock → Model access*,
  then give the service a **task role** with `bedrock:InvokeModel` permission (Express Mode creates standard ECS
  resources, so you can add the task role in the console or with the ECS API). Env vars:
  `LLM_PROVIDER=bedrock`, `LLM_MODEL=<model id or inference profile>`, `EMBEDDINGS_PROVIDER=bedrock`,
  `EMBEDDINGS_MODEL=amazon.titan-embed-text-v2:0`.
- **Option B – OpenAI:** `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`. In real projects store the key in
  **AWS Secrets Manager**, never in plain environment variables or in Git.

## 4. Create the service (one command)
```bash
aws ecs create-express-gateway-service \
  --execution-role-arn arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole \
  --infrastructure-role-arn arn:aws:iam::$ACCOUNT_ID:role/ecsInfrastructureRoleForExpressServices \
  --primary-container '{"image":"'$ACCOUNT_ID'.dkr.ecr.'$AWS_REGION'.amazonaws.com/'$REPO':latest","containerPort":8000,
     "environment":[{"name":"LLM_PROVIDER","value":"bedrock"},{"name":"LLM_MODEL","value":"<your-model-id>"},
                    {"name":"EMBEDDINGS_PROVIDER","value":"bedrock"},{"name":"EMBEDDINGS_MODEL","value":"amazon.titan-embed-text-v2:0"},
                    {"name":"AWS_REGION","value":"'$AWS_REGION'"}]}' \
  --service-name plant-ops-copilot \
  --cpu 1 --memory 2 \
  --health-check-path "/health" \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":3}' \
  --monitor-resources
```
When the status is `ACTIVE`, open `https://<service-name>.ecs.<region>.on.aws/docs`.

## 5. Observe
```bash
aws ecs describe-express-gateway-service --service-arn <SERVICE_ARN>
```
Logs go to CloudWatch Logs automatically. Look for the `Copilot ready` and `Indexed N chunks` lines.

## 6. Clean up (avoid charges)
Delete the service from the ECS console (*Express Mode → service → Delete*, or the matching
`aws ecs delete-...` command shown by `aws ecs help`), then the ECR repository:
```bash
aws ecr delete-repository --repository-name $REPO --force --region $AWS_REGION
```

## Next steps towards production
| Concern | This demo | Production |
|---|---|---|
| Vector DB | Chroma inside each task (rebuilt on start) | Amazon OpenSearch Serverless (vector) or Aurora PostgreSQL + pgvector |
| Conversation memory | In-memory per task | Shared store (PostgreSQL / DynamoDB / Redis) so any replica continues a session |
| Secrets | Env vars | AWS Secrets Manager |
| Ingestion | At startup | Event-driven: S3 upload → SQS → indexing worker |
| Observability | CloudWatch logs | + traces of every LLM call (LangSmith, Langfuse or OpenTelemetry), cost per request |
| Access | Public URL | Auth (Cognito / API keys), WAF, rate limiting |
