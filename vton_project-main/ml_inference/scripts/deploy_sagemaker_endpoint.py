#!/usr/bin/env python
"""
deploy_sagemaker_endpoint.py — Create/update the DCI-VTON SageMaker Async
Inference endpoint via boto3.

This script is idempotent: each step checks for an existing resource and
updates/skips instead of failing. It does NOT build the Docker image or
the model.tar.gz — run build_and_push.sh and package_model.sh (or .ps1)
first and pass their output URIs here.

Usage:
    python deploy_sagemaker_endpoint.py \\
        --role-arn arn:aws:iam::960583974175:role/SageMakerVTONExecutionRole \\
        --image-uri 960583974175.dkr.ecr.us-east-1.amazonaws.com/dci-vton-sagemaker:latest \\
        --model-data-url s3://dci-vton-artifacts-960583974175/dci-vton/model/model.tar.gz \\
        --s3-output-path s3://dci-vton-artifacts-960583974175/dci-vton/async-output/ \\
        --region us-east-1 \\
        --instance-type ml.g5.xlarge

Add --sns-success-topic / --sns-error-topic to get completion notifications
(recommended so the FastAPI/Celery worker doesn't have to poll S3 blindly).
"""
from __future__ import annotations

import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", required=True)
    p.add_argument("--role-arn", required=True, help="IAM execution role ARN for the SageMaker model")
    p.add_argument("--image-uri", required=True, help="ECR image URI from build_and_push.sh")
    p.add_argument("--model-data-url", required=True, help="S3 URI of model.tar.gz from package_model.sh")
    p.add_argument("--s3-output-path", required=True, help="S3 prefix where async results are written")
    p.add_argument("--model-name", default="dci-vton-model")
    p.add_argument("--endpoint-config-name", default="dci-vton-async-config")
    p.add_argument("--endpoint-name", default="dci-vton-async-endpoint")
    p.add_argument("--instance-type", default="ml.g5.xlarge",
                   help="ml.g5.xlarge (A10G, recommended) or ml.g4dn.xlarge (T4, budget)")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--max-concurrent-invocations", type=int, default=2)
    p.add_argument("--min-capacity", type=int, default=0, help="0 = scale-to-zero between bursts")
    p.add_argument("--max-capacity", type=int, default=2)
    p.add_argument("--sns-success-topic", default=None)
    p.add_argument("--sns-error-topic", default=None)
    p.add_argument("--model-download-timeout", type=int, default=900,
                   help="Seconds allowed to download model.tar.gz (large ~3.7GB checkpoint set)")
    p.add_argument("--container-startup-timeout", type=int, default=900,
                   help="Seconds allowed for model_fn() to finish loading all models")
    p.add_argument("--skip-autoscaling", action="store_true",
                   help="Skip registering Application Auto Scaling (e.g. for a first smoke-test deploy)")
    return p.parse_args()


def create_or_reuse_model(sm, args) -> str:
    try:
        sm.describe_model(ModelName=args.model_name)
        print(f"[model] '{args.model_name}' already exists — reusing. "
              f"Delete it first if you changed the image/model data.")
        return args.model_name
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    print(f"[model] creating '{args.model_name}'")
    sm.create_model(
        ModelName=args.model_name,
        PrimaryContainer={
            "Image": args.image_uri,
            "ModelDataUrl": args.model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": "600",
            },
        },
        ExecutionRoleArn=args.role_arn,
    )
    return args.model_name


def create_or_update_endpoint_config(sm, args) -> str:
    async_config = {
        "OutputConfig": {
            "S3OutputPath": args.s3_output_path,
        },
        "ClientConfig": {
            "MaxConcurrentInvocationsPerInstance": args.max_concurrent_invocations,
        },
    }
    notif = {}
    if args.sns_success_topic:
        notif["SuccessTopic"] = args.sns_success_topic
    if args.sns_error_topic:
        notif["ErrorTopic"] = args.sns_error_topic
    if notif:
        async_config["OutputConfig"]["NotificationConfig"] = notif

    config_name = args.endpoint_config_name
    try:
        sm.describe_endpoint_config(EndpointConfigName=config_name)
        # Endpoint configs are immutable — version the name instead of failing.
        config_name = f"{args.endpoint_config_name}-{int(time.time())}"
        print(f"[endpoint-config] '{args.endpoint_config_name}' already exists — "
              f"creating versioned config '{config_name}' instead")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    print(f"[endpoint-config] creating '{config_name}'")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": args.model_name,
            "InstanceType": args.instance_type,
            "InitialInstanceCount": args.instance_count,
            "ModelDataDownloadTimeoutInSeconds": args.model_download_timeout,
            "ContainerStartupHealthCheckTimeoutInSeconds": args.container_startup_timeout,
        }],
        AsyncInferenceConfig=async_config,
    )
    return config_name


def create_or_update_endpoint(sm, args, config_name: str) -> None:
    try:
        sm.describe_endpoint(EndpointName=args.endpoint_name)
        print(f"[endpoint] '{args.endpoint_name}' exists — updating to config '{config_name}'")
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
        waiter = sm.get_waiter("endpoint_in_service")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise
        print(f"[endpoint] creating '{args.endpoint_name}'")
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
        waiter = sm.get_waiter("endpoint_in_service")

    print("[endpoint] waiting for InService (this can take several minutes — "
          "model_fn loads ~3.7GB of checkpoints)...")
    waiter.wait(EndpointName=args.endpoint_name, WaiterConfig={"Delay": 15, "MaxAttempts": 120})
    print(f"[endpoint] '{args.endpoint_name}' is InService.")


def register_autoscaling(args) -> None:
    """Scale-to-zero / scale-on-queue-backlog for the async endpoint variant."""
    aas = boto3.client("application-autoscaling", region_name=args.region)
    resource_id = f"endpoint/{args.endpoint_name}/variant/AllTraffic"
    namespace = "sagemaker"
    dimension = "sagemaker:variant:DesiredInstanceCount"

    print(f"[autoscaling] registering scalable target ({args.min_capacity}-{args.max_capacity})")
    aas.register_scalable_target(
        ServiceNamespace=namespace,
        ResourceId=resource_id,
        ScalableDimension=dimension,
        MinCapacity=args.min_capacity,
        MaxCapacity=args.max_capacity,
    )

    print("[autoscaling] applying target-tracking policy on ApproximateBacklogSizePerInstance")
    aas.put_scaling_policy(
        PolicyName="dci-vton-backlog-scaling",
        ServiceNamespace=namespace,
        ResourceId=resource_id,
        ScalableDimension=dimension,
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 5.0,
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": args.endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    print("[autoscaling] done — endpoint can scale to 0 instances when idle.")


def main() -> int:
    args = parse_args()
    sm = boto3.client("sagemaker", region_name=args.region)

    create_or_reuse_model(sm, args)
    config_name = create_or_update_endpoint_config(sm, args)
    create_or_update_endpoint(sm, args, config_name)

    if not args.skip_autoscaling:
        register_autoscaling(args)
    else:
        print("[autoscaling] skipped (--skip-autoscaling)")

    print("\nDone. Set this in your FastAPI .env:")
    print(f"  SAGEMAKER_ENDPOINT_NAME={args.endpoint_name}")
    print(f"  SAGEMAKER_REGION={args.region}")
    print(f"  SAGEMAKER_ASYNC_S3_OUTPUT_PATH={args.s3_output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
