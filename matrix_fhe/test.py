import random
import functools
import time

from matrix_fhe import (
    generate_key,
    encrypt,
    decrypt,
    binary_encrypt,
    binary_decrypt,
    encoded_add,
    multi_add,
    _xor,
    _and,
    two_of_three,
    three_to_two,
)


DIMENSION = 5
PRECISION = 96

args = [{"dimension": DIMENSION, "precision": PRECISION}]


def testcase(test_name_or_func=None, args=None):
    test_name = None

    def decorator(function):
        @functools.wraps(function)
        def wrapper():
            name = test_name if test_name is not None else function.__name__
            if args is None:
                print(f"Running {name}")
                start_time = time.time()
                function()
                end_time = time.time()
                duration = end_time - start_time
                print(f"{name} passed in {duration:.4f}s")
            else:
                total_start_time = time.time()
                for i, arg_set in enumerate(args):
                    test_instance_name = f"{name} (case {i + 1})"
                    print(f"Running {test_instance_name}")
                    start_time = time.time()
                    if isinstance(arg_set, (list, tuple)):
                        function(*arg_set)
                    elif isinstance(arg_set, dict):
                        function(**arg_set)
                    else:
                        function(arg_set)
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"{test_instance_name} passed in {duration:.4f}s")
                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                print(
                    f"{name} completed all {len(args)} cases in {total_duration:.4f}s"
                )

        return wrapper

    if callable(test_name_or_func):
        return decorator(test_name_or_func)
    else:
        test_name = test_name_or_func
        return decorator


@testcase("basic_test", args=args)
def basic_test(*, dimension, precision):
    for _ in range(10):
        key = generate_key(dimension, precision)
        x = random.randrange(2)
        ct = encrypt(key, x, precision)
        pt = decrypt(key, ct, precision)

        assert pt == x


@testcase(
    "addition_multiplication_test",
    args=args,
)
def addition_multiplication_test(*, dimension, precision):
    for i in range(10):
        key = generate_key(dimension, precision)
        v1 = random.randrange(2)
        v2 = random.randrange(2)
        ct1 = encrypt(key, v1, precision)
        ct2 = encrypt(key, v2, precision)
        ct3 = _xor(ct1, ct2, precision)
        ct4 = _and(ct1, ct2, precision)
        ct5 = ct4

        for i in range(5):
            ct5 = _and(ct5, ct5, precision)

        assert decrypt(key, ct3, precision) == v1 ^ v2
        assert decrypt(key, ct4, precision) == v1 * v2
        assert decrypt(key, ct5, precision) == v1 * v1 * v2

    for i in range(8):
        a, b, c = i % 2, (i // 2) % 2, i // 4
        cta, ctb, ctc = (encrypt(key, x, precision) for x in (a, b, c))
        cto = two_of_three(cta, ctb, ctc, precision)

        assert decrypt(key, cto, precision) == (1 if a + b + c >= 2 else 0)


@testcase("simple_addition_test", args=args)
def simple_addition_test(*, dimension, precision):
    key = generate_key(dimension, precision)

    forty_two = binary_encrypt(key, 42, 8, precision)
    sixty_nine = binary_encrypt(key, 69, 8, precision)

    assert binary_decrypt(key, forty_two, precision) == 42
    assert binary_decrypt(key, sixty_nine, precision) == 69

    one_one_one = encoded_add(forty_two, sixty_nine, precision)

    assert binary_decrypt(key, one_one_one[:8], precision) == 111


@testcase("complete_addition_test", args=args)
def complete_addition_test(*, dimension, precision):
    key = generate_key(dimension, precision)

    for a in range(4):
        ct1 = binary_encrypt(key, a, 2, precision)
        for b in range(4):
            ct2 = binary_encrypt(key, b, 2, precision)

            assert (
                binary_decrypt(key, encoded_add(ct1, ct2, precision), precision)
                == a + b
            )

            for c in range(4):
                ct3 = binary_encrypt(key, c, 2, precision)
                x, y = three_to_two(ct1, ct2, ct3, precision)

                assert (
                    binary_decrypt(key, x, precision)
                    + binary_decrypt(key, y, precision)
                    == a + b + c
                )


@testcase("three_item_addition_test", args=args)
def three_item_addition_test(*, dimension, precision):
    k = generate_key(dimension, precision)
    forty_two = binary_encrypt(k, 42, 8, precision)
    sixty_nine = binary_encrypt(k, 69, 8, precision)
    one_one_one = encoded_add(forty_two, sixty_nine, precision)
    x, y = three_to_two(forty_two, sixty_nine, one_one_one, precision)
    assert binary_decrypt(k, x, precision) + binary_decrypt(k, y, precision) == 222
    two_two_two = multi_add([forty_two, sixty_nine, one_one_one], precision, bits=8)
    assert binary_decrypt(k, two_two_two[:8], precision) == 222


@testcase("less_simple_addition_test", args=args)
def less_simple_addition_test(*, dimension, precision):
    k = generate_key(dimension, precision)
    values = [random.randrange(1000) for i in range(8)]
    ciphertexts = [binary_encrypt(k, v, 11, precision) for v in values]
    sum_ciphertext = multi_add(ciphertexts, precision, bits=14)
    assert binary_decrypt(k, sum_ciphertext, precision) == sum(values)


def test():
    basic_test()

    addition_multiplication_test()

    simple_addition_test()

    complete_addition_test()

    three_item_addition_test()

    less_simple_addition_test()


if __name__ == "__main__":
    test()
