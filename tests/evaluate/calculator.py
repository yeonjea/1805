import numpy as np 

def cosine_similarity(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("lenth is not matched")
    
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    dot_product = np.dot(v1, v2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    similarity = dot_product / (norm1 * norm2)

    return similarity

def calculate_aie(vector1, vector2):
    # 벡터의 차의 절대값의 평균
    if len(vector1) != len(vector2):
        raise ValueError("lenth is not matched")

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    absolute_diff = np.abs(v1 - v2)

    mean_diff = np.mean(absolute_diff)
    aie = mean_diff
    return aie

def generate_logit_samples(num_samples, num_classes):
    # 샘플 수와 클래스 수에 따라 임의의 logit 값 생성
    logits = np.random.randn(num_samples, num_classes)
    return logits

if __name__ == "__main__":
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    similarity = cosine_similarity(vector1, vector2)
    print(f"cosine_similarity: {similarity:.3f}")

    vector1 = [1, 2, 3, 4, 5]
    vector2 = [4, 5, 6, 7, 8]
    aie = calculate_aie(vector1, vector2)
    print(f"calculate_aie: {aie:.2f}")

    num_samples = 1
    num_classes = 3
    logit_1 = generate_logit_samples(num_samples, num_classes)
    logit_2 = generate_logit_samples(num_samples, num_classes)
    aie = calculate_aie(logit_1, logit_2)
    print(f"calculate_aie: {aie:.2f}")
