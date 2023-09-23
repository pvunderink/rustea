use arrayvec::ArrayVec;

pub trait FromIteratorUnsafe<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self;
}

impl<T, const N: usize> FromIteratorUnsafe<T> for [T; N]
where
    T: Default + Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut res = [T::default(); N];
        for (it, elem) in res.iter_mut().zip(iter) {
            *it = elem
        }

        res
    }
}

impl<T, const N: usize> FromIteratorUnsafe<T> for ArrayVec<T, N>
where
    T: Default + Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        <ArrayVec<T, N> as FromIterator<T>>::from_iter(iter)
    }
}

pub trait CollectUnsafe {
    type Item;

    fn collect_unsafe<B>(self) -> B
    where
        B: FromIteratorUnsafe<Self::Item>,
        Self: Sized;
}

impl<I> CollectUnsafe for I
where
    I: Iterator,
{
    type Item = I::Item;

    fn collect_unsafe<B>(self) -> B
    where
        B: FromIteratorUnsafe<Self::Item>,
        Self: Sized,
    {
        B::from_iter(self)
    }
}
